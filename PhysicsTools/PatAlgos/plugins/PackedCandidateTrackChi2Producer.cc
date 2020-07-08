#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

namespace pat {

  class PackedCandidateTrackChi2Producer : public edm::stream::EDProducer<> {
    typedef edm::ValueMap<float> FloatMap;

  public:
    explicit PackedCandidateTrackChi2Producer(const edm::ParameterSet& iConfig)
        : candidateToken_(consumes<edm::View<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("candidates"))),
          trackToken_(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("trackCollection"))),
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
    ~PackedCandidateTrackChi2Producer() override{};

    void produce(edm::Event&, const edm::EventSetup&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::EDGetTokenT<edm::View<pat::PackedCandidate>> candidateToken_;
    edm::EDGetTokenT<edm::Association<reco::PFCandidateCollection>> candidate2PFToken_;
    edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> track2LostTrackToken_;
    edm::EDGetTokenT<edm::View<reco::Track>> trackToken_;
    bool doLostTracks_;
  };

}  // namespace pat

void pat::PackedCandidateTrackChi2Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<pat::PackedCandidate>> candidates;
  iEvent.getByToken(candidateToken_, candidates);

  edm::Handle<edm::Association<reco::PFCandidateCollection>> candidate2PF;
  if (!doLostTracks_) {
    iEvent.getByToken(candidate2PFToken_, candidate2PF);
  }

  edm::Handle<edm::Association<pat::PackedCandidateCollection>> tracks2LT;
  edm::Handle<edm::View<reco::Track>> trks;
  if (doLostTracks_) {
    iEvent.getByToken(track2LostTrackToken_, tracks2LT);
    iEvent.getByToken(trackToken_, trks);
  }

  const auto& nCand = candidates->size();
  std::vector<float> trkChi2Map(nCand, 0);

  for (size_t i = 0; i < nCand; i++) {
    const auto& cand = candidates->refAt(i);
    float nChi2 = 0;

    // ignore neutral candidates or without track
    if (cand->charge() == 0 || !cand->hasTrackDetails())
      continue;

    if (doLostTracks_) {  //for Lost tracks we don't have references to PFCands, so we must loop over tracks and check keys...
      for (size_t j = 0; j < trks->size(); j++) {
        const auto& trk = trks->refAt(j);
        const auto& lostTrack = (*tracks2LT)[trk];
        if (lostTrack.isNonnull() && (cand.id() == lostTrack.id()) && (cand.key() == lostTrack.key())) {
          nChi2 = trk->normalizedChi2();
          break;
        }
      }
    } else {  //for the regular PackedPFCands we have direct references...
      const auto& candTrack = (*candidate2PF)[cand]->trackRef();
      nChi2 = candTrack->normalizedChi2();
    }

    trkChi2Map.at(i) = nChi2;
  }

  // fill the value maps
  std::unique_ptr<FloatMap> valueMap(new FloatMap());
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
