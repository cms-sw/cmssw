#include "CommonTools/RecoAlgos/interface/TrackWithVertexSelector.h"
#include "FWCore/Utilities/interface/isFinite.h"
//
// constructors and destructor
//

namespace {
  constexpr float fakeBeamSpotTimeWidth = 0.175f;
}

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
  timesToken_(iC.consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("timesTag"))),
  timeResosToken_(iC.consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("timeResosTag"))),
  vtxFallback_(iConfig.getParameter<bool>("vtxFallback")),
  zetaVtx_(iConfig.getParameter<double>("zetaVtx")),
  rhoVtx_(iConfig.getParameter<double>("rhoVtx")),
  nSigmaDtVertex_(iConfig.getParameter<double>("nSigmaDtVertex")) {
}

TrackWithVertexSelector::~TrackWithVertexSelector() {  }


void TrackWithVertexSelector::init(const edm::Event & event) {
    edm::Handle<reco::VertexCollection> hVtx;
    event.getByToken(vertexToken_, hVtx);
    vcoll_ = hVtx.product();

    edm::Handle<edm::ValueMap<float> > hTimes;
    event.getByToken(timesToken_, hTimes);
    timescoll_ = hTimes.isValid() ? hTimes.product() : nullptr;

    edm::Handle<edm::ValueMap<float> > hTimeResos;
    event.getByToken(timeResosToken_, hTimeResos);
    timeresoscoll_ = hTimeResos.isValid() ? hTimeResos.product() : nullptr;
}

bool TrackWithVertexSelector::testTrack(const reco::Track &t) const {
  using std::abs;
  if ((t.numberOfValidHits() >= numberOfValidHits_) &&
      (static_cast<unsigned int>(t.hitPattern().numberOfValidPixelHits()) >= numberOfValidPixelHits_) &&
      (t.numberOfLostHits() <= numberOfLostHits_) &&
      (t.normalizedChi2()    <= normalizedChi2_) &&
      (t.ptError()/t.pt()*std::max(1.,t.normalizedChi2()) <= ptErrorCut_) &&
      (t.quality(t.qualityByName(quality_))) &&
      (t.pt()              >= ptMin_)      &&
      (t.pt()              <= ptMax_)      &&
      (abs(t.eta())   <= etaMax_)     &&
      (abs(t.eta())   >= etaMin_)     &&
      (abs(t.dz())    <= dzMax_)      &&
      (abs(t.d0())    <= d0Max_)  ) {
    return true;
  }
  return false;
}

bool TrackWithVertexSelector::testTrack(const reco::TrackRef &tref) const {
  return testTrack(*tref);
}

bool TrackWithVertexSelector::testVertices(const reco::Track& t, const reco::VertexCollection &vtxs) const {
  bool ok = false;
  if (!vtxs.empty()) {
    unsigned int tested = 1;
    for (reco::VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end();
	 it != ed; ++it) {
      if ( (std::abs(t.dxy(it->position())) < rhoVtx_) &&
           (std::abs(t.dz(it->position())) < zetaVtx_)  ) {
        ok = true; break;
      }
      if (tested++ >= nVertices_) break;
    }
  } else if (vtxFallback_) {
    return ( (std::abs(t.vertex().z()) < 15.9) && (t.vertex().Rho() < 0.2) );
  }
  return ok;
}

bool TrackWithVertexSelector::testVertices(const reco::TrackRef &tref, const reco::VertexCollection &vtxs) const {
  const auto& t = *tref;
  const bool timeAvailable = timescoll_ != nullptr && timeresoscoll_ != nullptr;
  bool ok = false;
  if (!vtxs.empty()) {
    unsigned int tested = 1;
    for (reco::VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end();
	 it != ed; ++it) {
      const bool useTime = timeAvailable && it->t() != 0.;
      float time = useTime ? (*timescoll_)[tref] : -1.f;
      float timeReso = useTime ? (*timeresoscoll_)[tref] : -1.f;
      timeReso = ( timeReso > 1e-6 ? timeReso : fakeBeamSpotTimeWidth );

      if( edm::isNotFinite(time) ) {
	time = 0.0;
	timeReso = 2.0*fakeBeamSpotTimeWidth;
      }

      const double vtxSigmaT2 = it->tError() * it->tError();
      const double vtxTrackErr = std::sqrt( vtxSigmaT2 + timeReso*timeReso );

      if ( (std::abs(t.dxy(it->position())) < rhoVtx_) &&
           (std::abs(t.dz(it->position())) < zetaVtx_) &&
	   ( !useTime || (std::abs(time - it->t())/vtxTrackErr < nSigmaDtVertex_) ) ) {
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

bool TrackWithVertexSelector::operator()(const reco::TrackRef &tref) const {
  if (!testTrack(tref)) return false;
  if (nVertices_ == 0) return true;
  return testVertices(tref, *vcoll_);
}

