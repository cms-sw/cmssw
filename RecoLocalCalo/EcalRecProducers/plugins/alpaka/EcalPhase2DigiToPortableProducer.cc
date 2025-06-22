#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiPhase2HostCollection.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiPhase2DeviceCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class EcalPhase2DigiToPortableProducer : public global::EDProducer<> {
  public:
    explicit EcalPhase2DigiToPortableProducer(edm::ParameterSet const &ps);
    ~EcalPhase2DigiToPortableProducer() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

    void produce(edm::StreamID sid, device::Event &event, device::EventSetup const &setup) const override;

  private:
    const edm::EDGetTokenT<EBDigiCollectionPh2> inputDigiToken_;
    const edm::EDPutTokenT<EcalDigiPhase2HostCollection> outputDigiHostToken_;
  };

  void EcalPhase2DigiToPortableProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("BarrelDigis", edm::InputTag("simEcalUnsuppressedDigis", ""));
    desc.add<std::string>("digisLabelEB", "ebDigis");

    descriptions.addWithDefaultLabel(desc);
  }

  EcalPhase2DigiToPortableProducer::EcalPhase2DigiToPortableProducer(edm::ParameterSet const &ps)
      : EDProducer(ps),
        inputDigiToken_{consumes(ps.getParameter<edm::InputTag>("BarrelDigis"))},
        outputDigiHostToken_{produces(ps.getParameter<std::string>("digisLabelEB"))} {}

  void EcalPhase2DigiToPortableProducer::produce(edm::StreamID sid,
                                                 device::Event &event,
                                                 device::EventSetup const &setup) const {
    //input data from event
    const auto &inputDigis = event.get(inputDigiToken_);

    const uint32_t size = inputDigis.size();

    //create host and device Digi collections of required size
    EcalDigiPhase2HostCollection digisHostColl{static_cast<int32_t>(size), event.queue()};
    auto digisHostCollView = digisHostColl.view();

    //iterate over digis
    uint32_t i = 0;
    for (const auto &inputDigi : inputDigis) {
      const unsigned int nSamples = inputDigi.size();
      //assign id to host collection
      digisHostCollView.id()[i] = inputDigi.id();
      if (nSamples > ecalPh2::sampleSize) {
        edm::LogError("size_mismatch") << "Number of input samples (" << nSamples
                                       << ") larger than the maximum sample size (" << ecalPh2::sampleSize
                                       << "). Ignoring the excess samples.";
      }
      //iterate over sample in digi, make sure the size of the input is not larger than the max sample size in Phase 2, if smaller set to 0
      for (unsigned int sample = 0; sample < ecalPh2::sampleSize; ++sample) {
        if (sample < nSamples) {
          //get samples from input digi
          EcalLiteDTUSample thisSample = inputDigi[sample];
          //assign adc data to host collection
          digisHostCollView.data()[i][sample] = thisSample.raw();
        } else {
          digisHostCollView.data()[i][sample] = 0;
        }
      }
      ++i;
    }
    digisHostCollView.size() = i;

    //emplace device collection in the event
    event.emplace(outputDigiHostToken_, std::move(digisHostColl));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(EcalPhase2DigiToPortableProducer);
