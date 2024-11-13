#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/LST/interface/LSTPhase2OTHitsInput.h"

class LSTPhase2OTHitsInputProducer : public edm::global::EDProducer<> {
public:
  explicit LSTPhase2OTHitsInputProducer(edm::ParameterSet const& iConfig);
  ~LSTPhase2OTHitsInputProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> phase2OTRecHitToken_;
  const edm::EDPutTokenT<LSTPhase2OTHitsInput> lstPhase2OTHitsInputPutToken_;
};

LSTPhase2OTHitsInputProducer::LSTPhase2OTHitsInputProducer(edm::ParameterSet const& iConfig)
    : phase2OTRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("phase2OTRecHits"))),
      lstPhase2OTHitsInputPutToken_(produces()) {}

void LSTPhase2OTHitsInputProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("phase2OTRecHits", edm::InputTag("siPhase2RecHits"));

  descriptions.addWithDefaultLabel(desc);
}

void LSTPhase2OTHitsInputProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Setup
  auto const& phase2OTHits = iEvent.get(phase2OTRecHitToken_);

  // Vector definitions
  std::vector<unsigned int> ph2_detId;
  ph2_detId.reserve(phase2OTHits.dataSize());
  std::vector<float> ph2_x;
  ph2_x.reserve(phase2OTHits.dataSize());
  std::vector<float> ph2_y;
  ph2_y.reserve(phase2OTHits.dataSize());
  std::vector<float> ph2_z;
  ph2_z.reserve(phase2OTHits.dataSize());
  std::vector<TrackingRecHit const*> ph2_hits;
  ph2_hits.reserve(phase2OTHits.dataSize());

  for (auto const& it : phase2OTHits) {
    const DetId hitId = it.detId();
    for (auto const& hit : it) {
      ph2_detId.push_back(hitId.rawId());
      ph2_x.push_back(hit.globalPosition().x());
      ph2_y.push_back(hit.globalPosition().y());
      ph2_z.push_back(hit.globalPosition().z());
      ph2_hits.push_back(&hit);
    }
  }

  LSTPhase2OTHitsInput phase2OTHitsInput(
      std::move(ph2_detId), std::move(ph2_x), std::move(ph2_y), std::move(ph2_z), std::move(ph2_hits));
  iEvent.emplace(lstPhase2OTHitsInputPutToken_, std::move(phase2OTHitsInput));
}

DEFINE_FWK_MODULE(LSTPhase2OTHitsInputProducer);
