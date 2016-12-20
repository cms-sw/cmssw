#ifndef CommonTools_RecoAlgos_RecoTrackSelectorBase_h
#define CommonTools_RecoAlgos_RecoTrackSelectorBase_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


class RecoTrackSelectorBase {
public:
  RecoTrackSelectorBase() {}
  RecoTrackSelectorBase(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC):
    ptMin_(cfg.getParameter<double>("ptMin")),
    minRapidity_(cfg.getParameter<double>("minRapidity")),
    maxRapidity_(cfg.getParameter<double>("maxRapidity")),
    tip_(cfg.getParameter<double>("tip")),
    lip_(cfg.getParameter<double>("lip")),
    maxChi2_(cfg.getParameter<double>("maxChi2")),
    minHit_(cfg.getParameter<int>("minHit")),
    minPixelHit_(cfg.getParameter<int>("minPixelHit")),
    minLayer_(cfg.getParameter<int>("minLayer")),
    min3DLayer_(cfg.getParameter<int>("min3DLayer")),
    usePV_(cfg.getParameter<bool>("usePV")),
    bsSrcToken_(iC.consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))) {
      if (usePV_)
        vertexToken_ = iC.consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertexTag"));
      for(const std::string& quality: cfg.getParameter<std::vector<std::string> >("quality"))
        quality_.push_back(reco::TrackBase::qualityByName(quality));
      for(const std::string& algorithm: cfg.getParameter<std::vector<std::string> >("algorithm"))
        algorithm_.push_back(reco::TrackBase::algoByName(algorithm));
      for(const std::string& algorithm: cfg.getParameter<std::vector<std::string> >("originalAlgorithm"))
        originalAlgorithm_.push_back(reco::TrackBase::algoByName(algorithm));
      for(const std::string& algorithm: cfg.getParameter<std::vector<std::string> >("algorithmMaskContains"))
        algorithmMask_.push_back(reco::TrackBase::algoByName(algorithm));
    }

  void init(const edm::Event& event, const edm::EventSetup& es) {
     edm::Handle<reco::BeamSpot> beamSpot;
     event.getByToken(bsSrcToken_,beamSpot);
     vertex_ = beamSpot->position();
     if (!usePV_) return;
     edm::Handle<reco::VertexCollection> hVtx;
     event.getByToken(vertexToken_, hVtx);
     if (hVtx->empty()) return;
     vertex_ = (*hVtx)[0].position();
  }

  bool operator()( const reco::Track & t) const {
    bool quality_ok = true;
    if (quality_.size()!=0) {
      quality_ok = false;
      for (unsigned int i = 0; i<quality_.size();++i) {
	if (t.quality(quality_[i])){
	  quality_ok = true;
	  break;
	}
      }
    }

    bool algo_ok = true;
    if (algorithm_.size()!=0) {
      if (std::find(algorithm_.begin(),algorithm_.end(),t.algo())==algorithm_.end()) algo_ok = false;
    }
    if (!originalAlgorithm_.empty() && algo_ok) {
      if (std::find(originalAlgorithm_.begin(), originalAlgorithm_.end(), t.originalAlgo()) == originalAlgorithm_.end()) algo_ok = false;
    }
    if(!algorithmMask_.empty() && algo_ok) {
      if(std::find_if(algorithmMask_.begin(), algorithmMask_.end(), [&](reco::TrackBase::TrackAlgorithm algo) -> bool { // for some reason I have to either explicitly give the return type, or use static_cast<bool>()
            return t.algoMask()[algo];
          }) == algorithmMask_.end()) algo_ok = false;
    }
    return
      (
       (algo_ok & quality_ok) &&
       t.hitPattern().numberOfValidHits() >= minHit_ &&
       t.hitPattern().numberOfValidPixelHits() >= minPixelHit_ &&
       t.hitPattern().trackerLayersWithMeasurement() >= minLayer_ &&
       t.hitPattern().pixelLayersWithMeasurement() +
       t.hitPattern().numberOfValidStripLayersWithMonoAndStereo() >= min3DLayer_ &&
       fabs(t.pt()) >= ptMin_ &&
       t.eta() >= minRapidity_ && t.eta() <= maxRapidity_ &&
       fabs(t.dxy(vertex_)) <= tip_ &&
       fabs(t.dsz(vertex_)) <= lip_  &&
       t.normalizedChi2()<=maxChi2_
      );
  }


private:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  double maxChi2_;
  int    minHit_;
  int    minPixelHit_;
  int    minLayer_;
  int    min3DLayer_;
  bool usePV_;

  edm::EDGetTokenT<reco::BeamSpot> bsSrcToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  std::vector<reco::TrackBase::TrackQuality> quality_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithm_;
  std::vector<reco::TrackBase::TrackAlgorithm> originalAlgorithm_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithmMask_;

  reco::Track::Point vertex_;
};

#endif
