#include "CommonTools/RecoAlgos/interface/TrackWithVertexSelector.h"
//
// constructors and destructor
//

TrackWithVertexSelector::TrackWithVertexSelector(const edm::ParameterSet& iConfig, edm::ConsumesCollector & iC) :
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
  ptErrorCut_(iConfig.getParameter<double>("ptErrorCut")),
  quality_(iConfig.getParameter<std::string>("quality")),
  nVertices_(iConfig.getParameter<bool>("useVtx") ? iConfig.getParameter<uint32_t>("nVertices") : 0),
  vertexToken_(iC.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexTag"))),
  vtxFallback_(iConfig.getParameter<bool>("vtxFallback")),
  zetaVtx_(iConfig.getParameter<double>("zetaVtx")),
  rhoVtx_(iConfig.getParameter<double>("rhoVtx")) {
}

TrackWithVertexSelector::~TrackWithVertexSelector() {  }


void TrackWithVertexSelector::init(const edm::Event & event) {
    edm::Handle<reco::VertexCollection> hVtx;
    event.getByToken(vertexToken_, hVtx);
    vcoll_ = hVtx.product();
}


bool TrackWithVertexSelector::testTrack(const reco::Track &t) const {
  using std::abs;
  if ((t.numberOfValidHits() >= numberOfValidHits_) &&
      (static_cast<unsigned int>(t.hitPattern().numberOfValidPixelHits()) >= numberOfValidPixelHits_) &&
      (t.numberOfLostHits() <= numberOfLostHits_) &&
      (t.normalizedChi2()    <= normalizedChi2_) &&
      (t.ptError()/t.pt()*std::max(1.,t.normalizedChi2()) <= ptErrorCut_) &&
      (t.quality(t.qualityByName(quality_))) &&
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
  if (vtxs.size() > 0) {
    unsigned int tested = 1;
    for (reco::VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end();
	 it != ed; ++it) {
      if ((std::abs(t.dxy(it->position())) < rhoVtx_) &&
          (std::abs(t.dz(it->position())) < zetaVtx_)) {
        ok = true; break;
      }
      if (tested++ >= nVertices_) break;
    }
  } else if (vtxFallback_) {
    return ( (std::abs(t.vertex().z()) < 15.9) && (t.vertex().Rho() < 0.2) );
  }
  return ok;
}

bool TrackWithVertexSelector::operator()(const reco::Track &t) const {
  if (!testTrack(t)) return false;
  if (nVertices_ == 0) return true;
  return testVertices(t, *vcoll_);
}

