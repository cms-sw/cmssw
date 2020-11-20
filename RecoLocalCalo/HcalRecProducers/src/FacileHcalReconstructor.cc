#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class FacileHcalReconstructor : public SonicEDProducer<TritonClient> {
public:
  explicit FacileHcalReconstructor(edm::ParameterSet const& cfg)
      : SonicEDProducer<TritonClient>(cfg),
        fChannelInfoName_(cfg.getParameter<edm::InputTag>("ChannelInfoName")),
        fTokChannelInfo_(consumes<HBHEChannelInfoCollection>(fChannelInfoName_)),
        htopoToken_(esConsumes<HcalTopology, HcalRecNumberingRecord>()) {
    produces<HBHERecHitCollection>();
    setDebugName("FacileHcalReconstructor");
  }

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    edm::Handle<HBHEChannelInfoCollection> hChannelInfo;
    iEvent.getByToken(fTokChannelInfo_, hChannelInfo);

    const HcalTopology* htopo = &iSetup.getData(htopoToken_);

    auto& input1 = iInput.begin()->second;
    auto data1 = std::make_shared<TritonInput<float>>();
    data1->reserve(hChannelInfo->size());
    client_.setBatchSize(hChannelInfo->size());

    hcalIds_.clear();

    for (const auto& pChannel : *hChannelInfo) {
      std::vector<float> input;
      const HcalDetId pDetId = pChannel.id();
      hcalIds_.push_back(pDetId);

      //inputs for Facile: iphi, gain, raw[8], depth (categorical), ieta (categorical)
      input.push_back(pDetId.iphi());
      input.push_back(pChannel.tsGain(0.));
      for (unsigned int itTS = 0; itTS < pChannel.nSamples(); ++itTS) {
        input.push_back(pChannel.tsRawCharge(itTS));
      }

      for (int itDepth = 1; itDepth <= htopo->maxDepth(); itDepth++) {
        input.push_back(pDetId.depth() == itDepth);
      }

      for (int itIeta = 1; itIeta <= htopo->lastHERing(); itIeta++) {
        input.push_back(pDetId.ietaAbs() == itIeta);
      }

      data1->push_back(input);
    }

    input1.toServer(data1);
  }

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    std::unique_ptr<HBHERecHitCollection> out;
    out = std::make_unique<HBHERecHitCollection>();
    out->reserve(hcalIds_.size());

    const auto& output1 = iOutput.begin()->second;
    const auto& outputs = output1.fromServer<float>();
    for (std::size_t iB = 0; iB < hcalIds_.size(); iB++) {
      float rhE = outputs[iB][0];
      if (rhE < 0. or std::isnan(rhE) or std::isinf(rhE))
        rhE = 0;

      HBHERecHit rh = HBHERecHit(hcalIds_[iB], rhE, 0.f, 0.f);
      out->push_back(rh);
    }
    iEvent.put(std::move(out));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    desc.add<edm::InputTag>("ChannelInfoName");
    descriptions.add("FacileHcalReconstructor", desc);
  }

private:
  edm::InputTag fChannelInfoName_;
  edm::EDGetTokenT<HBHEChannelInfoCollection> fTokChannelInfo_;
  std::vector<HcalDetId> hcalIds_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> htopoToken_;
};

DEFINE_FWK_MODULE(FacileHcalReconstructor);
