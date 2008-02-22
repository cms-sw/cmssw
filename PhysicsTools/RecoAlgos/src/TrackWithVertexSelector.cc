#include "PhysicsTools/RecoAlgos/interface/TrackWithVertexSelector.h"
//
// constructors and destructor
//

TrackWithVertexSelector::TrackWithVertexSelector(const edm::ParameterSet& iConfig) :
  numberOfValidHits_(iConfig.getParameter<uint32_t>("numberOfValidHits")),
  numberOfValidPixelHits_(iConfig.getParameter<uint32_t>("numberOfValidPixelHits")),
  numberOfLostHits_(iConfig.getParameter<uint32_t>("numberOfLostHits")),
  normalizedChi2_(iConfig.getParameter<double>("normalizedChi2")),
  ptMin_(iConfig.getParameter<double>("ptMin")),
  ptMax_(iConfig.getParameter<double>("ptMax")),
  etaMin_(iConfig.getParameter<double>("etaMin")),
  etaMax_(iConfig.getParameter<double>("etaMax")),
  dzMax_(iConfig.getParameter<double>("dzMax")),
  d0Max_(iConfig.getParameter<double>("d0Max")),
  nVertices_(iConfig.getParameter<bool>("useVtx") ? iConfig.getParameter<uint32_t>("nVertices") : 0),
  vertexTag_(iConfig.getParameter<edm::InputTag>("vertexTag")),
  vtxFallback_(iConfig.getParameter<bool>("vtxFallback")),
  zetaVtx_(iConfig.getParameter<double>("zetaVtx")),
  rhoVtx_(iConfig.getParameter<double>("rhoVtx")) {
} 

TrackWithVertexSelector::~TrackWithVertexSelector() {  }

bool TrackWithVertexSelector::testTrack(const reco::Track &t) const {
  using std::abs;
  if ((t.numberOfValidHits() >= numberOfValidHits_) &&
      (static_cast<unsigned int>(t.hitPattern().numberOfValidPixelHits()) >= numberOfValidPixelHits_) &&
      (t.numberOfLostHits() <= numberOfLostHits_) &&
      (t.normalizedChi2()    <= normalizedChi2_) &&
      (t.pt()         >= ptMin_)      &&
      (t.pt()         <= ptMax_)      &&
      (abs(t.eta())   <= etaMax_)     &&
      (abs(t.eta())   >= etaMin_)     &&
      (abs(t.dz())    <= dzMax_)      &&
      (abs(t.d0())    <= d0Max_)  ) {
    return true;
  }
  return false;
}

bool TrackWithVertexSelector::testVertices(const reco::Track &t, const reco::VertexCollection &vtxs) const {
  bool ok = false;
  const Point &pca = t.vertex();
  if (vtxs.size() > 0) {
    unsigned int tested = 1;
    for (reco::VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end();
	 it != ed; ++it) {
      if (testPoint(pca, it->position())) { ok = true; break; }
      if (tested++ >= nVertices_) break;
    }
  } else if (vtxFallback_) {
    return ( (std::abs(pca.z()) < 15.9) && (pca.Rho() < 0.2) );
  }
  return ok;
} 

bool TrackWithVertexSelector::operator()(const reco::Track &t, const edm::Event &evt) const {
  using std::abs;
  if (!testTrack(t)) return false;
  if (nVertices_ == 0) return true;
  edm::Handle<reco::VertexCollection> hVtx;
  evt.getByLabel(vertexTag_, hVtx);
  return testVertices(t, *hVtx);
} 

bool TrackWithVertexSelector::operator()(const reco::Track &t, const reco::VertexCollection &vtxs) const {
  using std::abs;
  if (!testTrack(t)) return false;
  if (nVertices_ == 0) return true;
  return testVertices(t, vtxs);
}

bool TrackWithVertexSelector::testPoint(const Point &point, const Point &vtx) const {
  using std::abs;
  math::XYZVector d = point - vtx;
  return ((abs(d.z()) < zetaVtx_) && (abs(d.Rho()) < rhoVtx_));
}
