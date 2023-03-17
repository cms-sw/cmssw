#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataMerger.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/MuonSimInfo.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"

class MuonSpecialVariables : public edm::stream::EDProducer<> {
public:
  explicit MuonSpecialVariables(const edm::ParameterSet &iConfig)
      : muonSrc_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muonSrc"))),
        vertexSrc_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertexSrc"))),
        trkSrc_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trkSrc"))) {
    produces<std::vector<pat::Muon>>();
  }
  ~MuonSpecialVariables() override {}

  void setArbitratedTracker(pat::Muon &mu) const;
  void setInnerTrackDxyz(pat::Muon &mu, const reco::Vertex &vtx) const;
  void setTrkiso04(pat::Muon &mu, const std::vector<reco::Track> tracks) const;
  void setNSegements(pat::Muon &mu) const;

private:
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

  edm::EDGetToken muonSrc_;
  edm::EDGetToken vertexSrc_;
  edm::EDGetToken trkSrc_;
};

void MuonSpecialVariables::setArbitratedTracker(pat::Muon &mu) const {
  mu.addUserInt("isArbitratedTracker", muon::isGoodMuon(mu, muon::TrackerMuonArbitrated));
}

void MuonSpecialVariables::setInnerTrackDxyz(pat::Muon &mu, const reco::Vertex &vtx) const {
  float innerTrackDxy =
      mu.innerTrack().isNonnull() ? mu.innerTrack()->dxy(reco::TrackBase::Point(vtx.x(), vtx.y(), vtx.z())) : -999.9;
  float innerTrackDz =
      mu.innerTrack().isNonnull() ? mu.innerTrack()->dz(reco::TrackBase::Point(vtx.x(), vtx.y(), vtx.z())) : -999.9;
  mu.addUserFloat("innerTrackDxy", innerTrackDxy);
  mu.addUserFloat("innerTrackDz", innerTrackDz);
}

void MuonSpecialVariables::setTrkiso04(pat::Muon &mu, const std::vector<reco::Track> tracks) const {
  float energy = 0;
  for (const auto &trk : tracks) {
    if (deltaR(mu.eta(), mu.phi(), trk.eta(), trk.phi()) > 0.4)
      continue;
    energy += trk.pt();
  }
  float Trkiso04 = (energy - mu.pt()) / mu.pt();
  float relTrkiso4 = (Trkiso04 > 0) ? Trkiso04 : 0;
  mu.addUserFloat("relTrkiso4", relTrkiso4);
}

void MuonSpecialVariables::setNSegements(pat::Muon &mu) const {
  int nsegments = 0;
  for (auto &chamber : mu.matches()) {
    if (chamber.id.det() != DetId::Muon)
      continue;
    if (chamber.id.subdetId() != MuonSubdetId::DT && chamber.id.subdetId() != MuonSubdetId::CSC)
      continue;
    nsegments += chamber.segmentMatches.size();
  }
  mu.addUserInt("nsegments", nsegments);
}

void MuonSpecialVariables::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<std::vector<pat::Muon>> muonSrc;
  iEvent.getByToken(muonSrc_, muonSrc);

  edm::Handle<std::vector<reco::Vertex>> vertices;
  iEvent.getByToken(vertexSrc_, vertices);

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(trkSrc_, trkCollection);

  std::unique_ptr<std::vector<pat::Muon>> muonCollection(new std::vector<pat::Muon>(*muonSrc));

  std::unique_ptr<std::vector<pat::Muon>> out(new std::vector<pat::Muon>());

  out->reserve(muonCollection->size());

  for (unsigned int i = 0; i < muonCollection->size(); i++) {
    pat::Muon &mu = (*muonCollection).at(i);
    setArbitratedTracker(mu);
    int good_vertex = 0;
    for (auto &vtx : *vertices) {
      if (vtx.isFake() || !vtx.isValid())
        continue;
      setInnerTrackDxyz(mu, vtx);
      good_vertex = 1;
      break;
    }
    mu.addUserInt("isGoodVertex", good_vertex);
    setTrkiso04(mu, *trkCollection);
    setNSegements(mu);
    out->push_back(mu);
  }

  iEvent.put(std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonSpecialVariables);
