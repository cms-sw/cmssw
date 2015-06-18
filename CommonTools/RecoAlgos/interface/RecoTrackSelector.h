#ifndef RecoSelectors_RecoTrackSelector_h
#define RecoSelectors_RecoTrackSelector_h
/* \class RecoTrackSelector
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2013/06/05 14:49:09 $
 *  $Revision: 1.3.12.2 $
 *
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/Utilities/interface/InputTag.h"

class RecoTrackSelector {
 public:
  typedef reco::TrackCollection collection;
  typedef std::vector<const reco::Track *> container;
  typedef container::const_iterator const_iterator;

  /// Constructors
  RecoTrackSelector() {}
  RecoTrackSelector ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
    RecoTrackSelector( cfg, iC ) {}
  RecoTrackSelector ( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) :
    ptMin_(cfg.getParameter<double>("ptMin")),
    minRapidity_(cfg.getParameter<double>("minRapidity")),
    maxRapidity_(cfg.getParameter<double>("maxRapidity")),
    tip_(cfg.getParameter<double>("tip")),
    lip_(cfg.getParameter<double>("lip")),
    minHit_(cfg.getParameter<int>("minHit")),
    minPixelHit_(cfg.getParameter<int>("minPixelHit")),
    minLayer_(cfg.getParameter<int>("minLayer")),
    min3DLayer_(cfg.getParameter<int>("min3DLayer")),
    maxChi2_(cfg.getParameter<double>("maxChi2")),
    bsSrcToken_(iC.consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))),
    usePV_(cfg.getParameter<bool>("usePV")) {
      if (usePV_) vertexToken_ = iC.consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertexTag"));
      std::vector<std::string> quality = cfg.getParameter<std::vector<std::string> >("quality");
      for (unsigned int j=0;j<quality.size();j++) quality_.push_back(reco::TrackBase::qualityByName(quality[j]));
      std::vector<std::string> algorithm = cfg.getParameter<std::vector<std::string> >("algorithm");
      for (unsigned int j=0;j<algorithm.size();j++) algorithm_.push_back(reco::TrackBase::algoByName(algorithm[j]));
    }

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }


  void init(const edm::Event & event, const edm::EventSetup&es) {
     edm::Handle<reco::BeamSpot> beamSpot;
     event.getByToken(bsSrcToken_,beamSpot);
     vertex_ = beamSpot->position();
     if (!usePV_) return;
     edm::Handle<reco::VertexCollection> hVtx;
     event.getByToken(vertexToken_, hVtx);
     if (hVtx->empty()) return;
     vertex_ = (*hVtx)[0].position();
  }

  virtual void select( const edm::Handle<collection>& c, const edm::Event & event, const edm::EventSetup&es) {
    init(event,es);
    selected_.clear();
    for( reco::TrackCollection::const_iterator trk = c->begin();
         trk != c->end(); ++ trk )
      if ( operator()(*trk) ) {
	selected_.push_back( & * trk );
      }
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

  size_t size() const { return selected_.size(); }

 protected:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  int    minHit_;
  int    minPixelHit_;
  int    minLayer_;
  int    min3DLayer_;
  double maxChi2_;
  std::vector<reco::TrackBase::TrackQuality> quality_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithm_;
  edm::EDGetTokenT<reco::BeamSpot> bsSrcToken_;
  bool usePV_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  reco::Track::Point vertex_;
  container selected_;
  edm::EventID previousEvent;
};

#endif
