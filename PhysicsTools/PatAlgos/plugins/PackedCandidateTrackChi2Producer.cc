#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Math/interface/libminifloat.h"

namespace pat {

  class PackedCandidateTrackChi2Producer : public edm::stream::EDProducer<> {
    typedef edm::ValueMap<float> FloatMap;

  public:
    explicit PackedCandidateTrackChi2Producer(const edm::ParameterSet& iConfig)
        : candidateToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("candidates"))),
          trackToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackCollection"))),
          doLostTracks_(iConfig.getParameter<bool>("doLostTracks")) {
      if (doLostTracks_) {
        track2LostTrackToken_ = consumes<edm::Association<pat::PackedCandidateCollection>>(
            iConfig.getParameter<edm::InputTag>("candidates"));
      } else {
        candidate2PFToken_ =
            consumes<edm::Association<reco::PFCandidateCollection>>(iConfig.getParameter<edm::InputTag>("candidates"));
      }

      produces<FloatMap>();
    }

    void produce(edm::Event&, const edm::EventSetup&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const edm::EDGetTokenT<pat::PackedCandidateCollection> candidateToken_;
    edm::EDGetTokenT<edm::Association<reco::PFCandidateCollection>> candidate2PFToken_;
    edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> track2LostTrackToken_;
    const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
    const bool doLostTracks_;
    static const uint8_t roundingPrecision = 8;
  };

}  // namespace pat

void pat::PackedCandidateTrackChi2Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const candidates = iEvent.getHandle(candidateToken_);

  const edm::Association<reco::PFCandidateCollection>* candidate2PF = nullptr;
  if (!doLostTracks_) {
    candidate2PF = &iEvent.get(candidate2PFToken_);
  }

  const edm::Association<pat::PackedCandidateCollection>* tracks2LT = nullptr;
  edm::Handle<reco::TrackCollection> trks;
  if (doLostTracks_) {
    tracks2LT = &iEvent.get(track2LostTrackToken_);
    iEvent.getByToken(trackToken_, trks);
  }

  const auto nCand = candidates->size();
  std::vector<float> trkChi2Map(nCand, 0);

  if (doLostTracks_) {  //for Lost tracks we don't have references to PFCands, so we must loop over tracks and check keys...
    for (size_t i = 0; i < trks->size(); i++) {
      const auto& trk = reco::TrackRef(trks, i);
      const auto& lostTrack = (*tracks2LT)[trk];
      if (lostTrack.isNonnull()) {
        const float nChi2 = trk->normalizedChi2();
        trkChi2Map.at(lostTrack.key()) = MiniFloatConverter::reduceMantissaToNbitsRounding<roundingPrecision>(nChi2);
      }
    }
  } else {  //for the regular PackedPFCands we have direct references...
    for (size_t i = 0; i < nCand; i++) {
      const auto& cand = pat::PackedCandidateRef(candidates, i);

      // ignore neutral candidates or without track
      if (cand->charge() == 0 || !cand->hasTrackDetails())
        continue;

      const auto& candTrack = (*candidate2PF)[cand]->trackRef();

      if (candTrack.isNonnull()) {
        const float nChi2 = candTrack->normalizedChi2();

        trkChi2Map.at(i) = MiniFloatConverter::reduceMantissaToNbitsRounding<roundingPrecision>(nChi2);
      }
    }
  }

  // fill the value maps
  std::unique_ptr<FloatMap> valueMap = std::make_unique<FloatMap>();
  FloatMap::Filler filler(*valueMap);
  filler.insert(candidates, trkChi2Map.begin(), trkChi2Map.end());
  filler.fill();
  iEvent.put(std::move(valueMap), "");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void pat::PackedCandidateTrackChi2Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("candidates", edm::InputTag("packedPFCandidates"))
      ->setComment("packed candidate input collection");
  desc.add<edm::InputTag>("trackCollection", edm::InputTag("generalTracks"))->setComment("track input collection");
  desc.add<bool>("doLostTracks", false);
  descriptions.add("packedPFCandidateTrackChi2", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PackedCandidateTrackChi2Producer);
