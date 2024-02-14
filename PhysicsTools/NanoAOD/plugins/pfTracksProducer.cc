#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataMerger.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/MuonSimInfo.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"

class pfTracksProducer : public edm::stream::EDProducer<> {
public:
  explicit pfTracksProducer(const edm::ParameterSet& iConfig)
      : PFCands_(consumes<std::vector<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("PFCands"))),
        LostTracks_(consumes<std::vector<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("lostTracks"))),
        trkSelection_(iConfig.getParameter<std::string>("trkSelection")),
        HighPurity_(iConfig.getParameter<bool>("TrkHPurity")) {
    produces<reco::TrackCollection>();
  }
  ~pfTracksProducer() override {}

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetToken PFCands_;
  edm::EDGetToken LostTracks_;
  const StringCutObjectSelector<pat::PackedCandidate> trkSelection_;
  const bool HighPurity_;
};

void pfTracksProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<pat::PackedCandidate>> pfcands;
  iEvent.getByToken(PFCands_, pfcands);

  edm::Handle<std::vector<pat::PackedCandidate>> lostTracks;
  iEvent.getByToken(LostTracks_, lostTracks);

  std::unique_ptr<std::vector<reco::Track>> tracks(new std::vector<reco::Track>());
  tracks->reserve(pfcands->size() + lostTracks->size());

  for (const auto& container : {pfcands, lostTracks}) {
    for (const pat::PackedCandidate& trk : *container) {
      if (!trk.hasTrackDetails())
        continue;
      if (!trkSelection_(trk))
        continue;
      if (HighPurity_ && !trk.trackHighPurity())
        continue;
      tracks->emplace_back(*trk.bestTrack());
    }
  }

  iEvent.put(std::move(tracks));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(pfTracksProducer);
