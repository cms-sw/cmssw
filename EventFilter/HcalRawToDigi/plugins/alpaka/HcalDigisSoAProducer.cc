#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiHostCollection.h"
#include "DataFormats/HcalDigi/interface/alpaka/HcalDigiDeviceCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HcalDigisSoAProducer : public stream::EDProducer<> {
  public:
    explicit HcalDigisSoAProducer(edm::ParameterSet const& ps);
    ~HcalDigisSoAProducer() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, device::EventSetup const&) override;

  private:
    // input product tokens
    edm::EDGetTokenT<HBHEDigiCollection> hbheDigiToken_;
    edm::EDGetTokenT<QIE11DigiCollection> qie11DigiToken_;

    // type aliases
    using HostCollectionPhase1 = hcal::Phase1DigiHostCollection;
    using HostCollectionPhase0 = hcal::Phase0DigiHostCollection;

    // output product tokens
    edm::EDPutTokenT<HostCollectionPhase1> digisF01HEToken_;
    edm::EDPutTokenT<HostCollectionPhase0> digisF5HBToken_;
    edm::EDPutTokenT<HostCollectionPhase1> digisF3HBToken_;

    struct ConfigParameters {
      uint32_t maxChannelsF01HE, maxChannelsF5HB, maxChannelsF3HB;
    };
    ConfigParameters config_;
  };

  void HcalDigisSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("hbheDigisLabel", edm::InputTag("hcalDigis"));
    desc.add<edm::InputTag>("qie11DigiLabel", edm::InputTag("hcalDigis"));
    desc.add<std::string>("digisLabelF01HE", std::string{"f01HEDigis"});
    desc.add<std::string>("digisLabelF5HB", std::string{"f5HBDigis"});
    desc.add<std::string>("digisLabelF3HB", std::string{"f3HBDigis"});
    desc.add<uint32_t>("maxChannelsF01HE", 10000u);
    desc.add<uint32_t>("maxChannelsF5HB", 10000u);
    desc.add<uint32_t>("maxChannelsF3HB", 10000u);

    confDesc.addWithDefaultLabel(desc);
  }

  HcalDigisSoAProducer::HcalDigisSoAProducer(const edm::ParameterSet& ps)
      : EDProducer(ps),
        hbheDigiToken_{consumes(ps.getParameter<edm::InputTag>("hbheDigisLabel"))},
        qie11DigiToken_{consumes(ps.getParameter<edm::InputTag>("qie11DigiLabel"))},
        digisF01HEToken_{produces(ps.getParameter<std::string>("digisLabelF01HE"))},
        digisF5HBToken_{produces(ps.getParameter<std::string>("digisLabelF5HB"))},
        digisF3HBToken_{produces(ps.getParameter<std::string>("digisLabelF3HB"))} {
    config_.maxChannelsF01HE = ps.getParameter<uint32_t>("maxChannelsF01HE");
    config_.maxChannelsF5HB = ps.getParameter<uint32_t>("maxChannelsF5HB");
    config_.maxChannelsF3HB = ps.getParameter<uint32_t>("maxChannelsF3HB");
  }

  void HcalDigisSoAProducer::produce(device::Event& event, device::EventSetup const& setup) {
    const auto& hbheDigis = event.get(hbheDigiToken_);
    const auto& qie11Digis = event.get(qie11DigiToken_);

    //Get the number of samples in data from the first digi
    const int stride = HBHEDataFrame::MAXSAMPLES / 2 + 1;
    const int size = hbheDigis.size() * stride;  // number of channels * stride

    // stack host memory in the queue
    HostCollectionPhase0 hf5_(size, event.queue());

    // set SoA_Scalar;
    hf5_.view().stride() = stride;
    hf5_.view().size() = hbheDigis.size();

    for (unsigned int i = 0; i < hbheDigis.size(); ++i) {
      auto const hbhe = hbheDigis[i];
      auto const id = hbhe.id().rawId();
      auto const presamples = hbhe.presamples();
      uint16_t header_word = (1 << 15) | (0x5 << 12) | (0 << 10) | ((hbhe.sample(0).capid() & 0x3) << 8);

      auto hf5_vi = hf5_.view()[i];
      hf5_vi.ids() = id;
      hf5_vi.npresamples() = presamples;
      hf5_vi.data()[0] = header_word;
      //TODO:: get HEADER_WORDS/WORDS_PER_SAMPLE from DataFormat
      for (unsigned int i = 0; i < hf5_.view().stride() - 1; i++) {
        uint16_t s0 = (0 << 7) | (static_cast<uint8_t>(hbhe.sample(2 * i).adc()) & 0x7f);
        uint16_t s1 = (0 << 7) | (static_cast<uint8_t>(hbhe.sample(2 * i + 1).adc()) & 0x7f);
        uint16_t sample = (s1 << 8) | s0;
        hf5_vi.data()[i + 1] = sample;
      }
    }
    event.emplace(digisF5HBToken_, std::move(hf5_));

    if (qie11Digis.empty()) {
      event.emplace(digisF01HEToken_, 0, event.queue());
      event.emplace(digisF3HBToken_, 0, event.queue());

    } else {
      auto size_f1 = 0;
      auto size_f3 = 0;

      // count the size of the SOA;
      for (unsigned int i = 0; i < qie11Digis.size(); i++) {
        auto const digi = QIE11DataFrame{qie11Digis[i]};

        if (digi.flavor() == 0 or digi.flavor() == 1) {
          if (digi.detid().subdetId() == HcalEndcap) {
            size_f1++;
          }
        } else if (digi.flavor() == 3) {
          if (digi.detid().subdetId() == HcalBarrel) {
            size_f3++;
          }
        }
      }
      auto const nsamples = qie11Digis.samples();
      auto const stride01 = nsamples * QIE11DataFrame::WORDS_PER_SAMPLE + QIE11DataFrame::HEADER_WORDS;

      // stack host memory in the queue
      HostCollectionPhase1 hf1_(size_f1, event.queue());
      HostCollectionPhase1 hf3_(size_f3, event.queue());

      // set SoA_Scalar;
      hf1_.view().stride() = stride01;
      hf3_.view().stride() = stride01;

      unsigned int i_f1 = 0;  //counters for f1 digis
      unsigned int i_f3 = 0;  //counters for f3 digis

      for (unsigned int i = 0; i < qie11Digis.size(); i++) {
        auto const digi = QIE11DataFrame{qie11Digis[i]};
        assert(digi.samples() == qie11Digis.samples() && "collection nsamples must equal per digi samples");

        if (digi.flavor() == 0 or digi.flavor() == 1) {
          if (digi.detid().subdetId() != HcalEndcap)
            continue;
          auto hf01_vi = hf1_.view()[i_f1];

          hf01_vi.ids() = digi.detid().rawId();
          for (int hw = 0; hw < QIE11DataFrame::HEADER_WORDS + digi.samples(); hw++) {
            hf01_vi.data()[hw] = (qie11Digis[i][hw]);
          }
          i_f1++;
        } else if (digi.flavor() == 3) {
          if (digi.detid().subdetId() != HcalBarrel)
            continue;
          auto hf03_vi = hf3_.view()[i_f3];

          hf03_vi.ids() = digi.detid().rawId();

          for (int hw = 0; hw < QIE11DataFrame::HEADER_WORDS + digi.samples(); hw++) {
            hf03_vi.data()[hw] = (qie11Digis[i][hw]);
          }
          i_f3++;
        }
      }

      hf1_.view().size() = size_f1;
      hf3_.view().size() = size_f3;

      event.emplace(digisF01HEToken_, std::move(hf1_));
      event.emplace(digisF3HBToken_, std::move(hf3_));
    }
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HcalDigisSoAProducer);
