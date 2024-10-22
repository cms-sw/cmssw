#ifndef CommonTools_RecoAlgos_RecoTrackSelectorBase_h
#define CommonTools_RecoAlgos_RecoTrackSelectorBase_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"

class RecoTrackSelectorBase {
public:
  RecoTrackSelectorBase() {}
  RecoTrackSelectorBase(const edm::ParameterSet& cfg)
      : ptMin_(cfg.getParameter<double>("ptMin")),
        minRapidity_(cfg.getParameter<double>("minRapidity")),
        maxRapidity_(cfg.getParameter<double>("maxRapidity")),
        meanPhi_((cfg.getParameter<double>("minPhi") + cfg.getParameter<double>("maxPhi")) / 2.),
        rangePhi_((cfg.getParameter<double>("maxPhi") - cfg.getParameter<double>("minPhi")) / 2.),
        tip_(cfg.getParameter<double>("tip")),
        lip_(cfg.getParameter<double>("lip")),
        maxChi2_(cfg.getParameter<double>("maxChi2")),
        minHit_(cfg.getParameter<int>("minHit")),
        minPixelHit_(cfg.getParameter<int>("minPixelHit")),
        minLayer_(cfg.getParameter<int>("minLayer")),
        min3DLayer_(cfg.getParameter<int>("min3DLayer")),
        usePV_(false),
        invertRapidityCut_(cfg.getParameter<bool>("invertRapidityCut")) {
    const auto minPhi = cfg.getParameter<double>("minPhi");
    const auto maxPhi = cfg.getParameter<double>("maxPhi");
    if (minPhi >= maxPhi) {
      throw cms::Exception("Configuration")
          << "RecoTrackSelectorPhase: minPhi (" << minPhi << ") must be smaller than maxPhi (" << maxPhi
          << "). The range is constructed from minPhi to maxPhi around their average.";
    }
    if (minPhi >= M_PI) {
      throw cms::Exception("Configuration")
          << "RecoTrackSelectorPhase: minPhi (" << minPhi
          << ") must be smaller than PI. The range is constructed from minPhi to maxPhi around their average.";
    }
    if (maxPhi <= -M_PI) {
      throw cms::Exception("Configuration")
          << "RecoTrackSelectorPhase: maxPhi (" << maxPhi
          << ") must be larger than -PI. The range is constructed from minPhi to maxPhi around their average.";
    }

    for (const std::string& quality : cfg.getParameter<std::vector<std::string> >("quality"))
      quality_.push_back(reco::TrackBase::qualityByName(quality));
    for (const std::string& algorithm : cfg.getParameter<std::vector<std::string> >("algorithm"))
      algorithm_.push_back(reco::TrackBase::algoByName(algorithm));
    for (const std::string& algorithm : cfg.getParameter<std::vector<std::string> >("originalAlgorithm"))
      originalAlgorithm_.push_back(reco::TrackBase::algoByName(algorithm));
    for (const std::string& algorithm : cfg.getParameter<std::vector<std::string> >("algorithmMaskContains"))
      algorithmMask_.push_back(reco::TrackBase::algoByName(algorithm));
  }

  RecoTrackSelectorBase(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) : RecoTrackSelectorBase(cfg) {
    usePV_ = cfg.getParameter<bool>("usePV");
    bsSrcToken_ = iC.consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"));
    if (usePV_)
      vertexToken_ = iC.consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertexTag"));
  }

  void init(const edm::Event& event, const edm::EventSetup& es) {
    edm::Handle<reco::BeamSpot> beamSpot;
    event.getByToken(bsSrcToken_, beamSpot);
    vertex_ = beamSpot->position();
    if (!usePV_)
      return;
    edm::Handle<reco::VertexCollection> hVtx;
    event.getByToken(vertexToken_, hVtx);
    if (hVtx->empty())
      return;
    vertex_ = (*hVtx)[0].position();
  }

  bool operator()(const reco::TrackRef& tref) const { return (*this)(*tref); }

  bool operator()(const reco::Track& t) const { return (*this)(t, vertex_); }

  bool operator()(const reco::Track& t, const reco::Track::Point& vertex) const {
    bool quality_ok = true;
    if (!quality_.empty()) {
      quality_ok = false;
      for (unsigned int i = 0; i < quality_.size(); ++i) {
        if (t.quality(quality_[i])) {
          quality_ok = true;
          break;
        }
      }
    }

    bool algo_ok = true;
    if (!algorithm_.empty()) {
      if (std::find(algorithm_.begin(), algorithm_.end(), t.algo()) == algorithm_.end())
        algo_ok = false;
    }
    if (!originalAlgorithm_.empty() && algo_ok) {
      if (std::find(originalAlgorithm_.begin(), originalAlgorithm_.end(), t.originalAlgo()) == originalAlgorithm_.end())
        algo_ok = false;
    }
    if (!algorithmMask_.empty() && algo_ok) {
      if (std::find_if(algorithmMask_.begin(),
                       algorithmMask_.end(),
                       // for some reason I have to either explicitly give the return type, or use static_cast<bool>()
                       [&](reco::TrackBase::TrackAlgorithm algo) -> bool { return t.algoMask()[algo]; }) ==
          algorithmMask_.end())
        algo_ok = false;
    }

    const auto dphi = deltaPhi(t.phi(), meanPhi_);

    auto etaOk = [&](const reco::Track& p) -> bool {
      float eta = p.eta();
      if (!invertRapidityCut_)
        return (eta >= minRapidity_) && (eta <= maxRapidity_);
      else
        return (eta < minRapidity_ || eta > maxRapidity_);
    };

    return ((algo_ok & quality_ok) && t.hitPattern().numberOfValidHits() >= minHit_ &&
            t.hitPattern().numberOfValidPixelHits() >= minPixelHit_ &&
            t.hitPattern().trackerLayersWithMeasurement() >= minLayer_ &&
            t.hitPattern().pixelLayersWithMeasurement() + t.hitPattern().numberOfValidStripLayersWithMonoAndStereo() >=
                min3DLayer_ &&
            fabs(t.pt()) >= ptMin_ && etaOk(t) && dphi >= -rangePhi_ && dphi <= rangePhi_ &&
            fabs(t.dxy(vertex)) <= tip_ && fabs(t.dsz(vertex)) <= lip_ && t.normalizedChi2() <= maxChi2_);
  }

private:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double meanPhi_;
  double rangePhi_;
  double tip_;
  double lip_;
  double maxChi2_;
  int minHit_;
  int minPixelHit_;
  int minLayer_;
  int min3DLayer_;
  bool usePV_;
  bool invertRapidityCut_;

  edm::EDGetTokenT<reco::BeamSpot> bsSrcToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  std::vector<reco::TrackBase::TrackQuality> quality_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithm_;
  std::vector<reco::TrackBase::TrackAlgorithm> originalAlgorithm_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithmMask_;

  reco::Track::Point vertex_;
};

#endif
