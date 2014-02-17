#ifndef RecoSelectors_RecoTrackSelector_h
#define RecoSelectors_RecoTrackSelector_h
/* \class RecoTrackSelector
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2010/02/11 00:10:50 $
 *  $Revision: 1.3 $
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Utilities/interface/InputTag.h"

class RecoTrackSelector {
 public:
  typedef reco::TrackCollection collection;
  typedef std::vector<const reco::Track *> container;
  typedef container::const_iterator const_iterator;

  /// Constructors
  RecoTrackSelector() {}
  RecoTrackSelector ( const edm::ParameterSet & cfg ) :
    ptMin_(cfg.getParameter<double>("ptMin")),
    minRapidity_(cfg.getParameter<double>("minRapidity")),
    maxRapidity_(cfg.getParameter<double>("maxRapidity")),
    tip_(cfg.getParameter<double>("tip")),
    lip_(cfg.getParameter<double>("lip")),
    minHit_(cfg.getParameter<int>("minHit")),
    min3DHit_(cfg.getParameter<int>("min3DHit")),
    maxChi2_(cfg.getParameter<double>("maxChi2")),
    bsSrc_(cfg.getParameter<edm::InputTag>("beamSpot")) 
    {
      std::vector<std::string> quality = cfg.getParameter<std::vector<std::string> >("quality");
      for (unsigned int j=0;j<quality.size();j++) quality_.push_back(reco::TrackBase::qualityByName(quality[j]));
      std::vector<std::string> algorithm = cfg.getParameter<std::vector<std::string> >("algorithm");
      for (unsigned int j=0;j<algorithm.size();j++) algorithm_.push_back(reco::TrackBase::algoByName(algorithm[j]));
    }
  
  RecoTrackSelector ( double ptMin, double minRapidity, double maxRapidity,
		      double tip, double lip, int minHit, int min3DHit, double maxChi2, 
    		      std::vector<std::string> quality , std::vector<std::string> algorithm ) :
    ptMin_( ptMin ), minRapidity_( minRapidity ), maxRapidity_( maxRapidity ),
    tip_( tip ), lip_( lip ), minHit_( minHit ), min3DHit_( min3DHit), maxChi2_( maxChi2 ) 
    { 
      for (unsigned int j=0;j<quality.size();j++) quality_.push_back(reco::TrackBase::qualityByName(quality[j]));
      for (unsigned int j=0;j<algorithm.size();j++) algorithm_.push_back(reco::TrackBase::algoByName(algorithm[j]));
    }

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  
  void select( const edm::Handle<collection>& c, const edm::Event & event, const edm::EventSetup&) {
    selected_.clear();
    edm::Handle<reco::BeamSpot> beamSpot;
    event.getByLabel(bsSrc_,beamSpot); 
    for( reco::TrackCollection::const_iterator trk = c->begin(); 
         trk != c->end(); ++ trk )
      if ( operator()(*trk,beamSpot.product()) ) {
	selected_.push_back( & * trk );
      }
  }

  /// Operator() performs the selection: e.g. if (recoTrackSelector(track)) {...}
  bool operator()( const reco::Track & t, const reco::BeamSpot* bs) {
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
      (t.hitPattern().trackerLayersWithMeasurement() >= minHit_ &&
       t.hitPattern().pixelLayersWithMeasurement() +
       t.hitPattern().numberOfValidStripLayersWithMonoAndStereo() >= min3DHit_ &&
       fabs(t.pt()) >= ptMin_ &&
       t.eta() >= minRapidity_ && t.eta() <= maxRapidity_ &&
       fabs(t.dxy(bs->position())) <= tip_ &&
       fabs(t.dsz(bs->position())) <= lip_  &&
       t.normalizedChi2()<=maxChi2_ &&
       quality_ok &&
       algo_ok);
  }

  size_t size() const { return selected_.size(); }
  
 protected:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  int    minHit_;
  int    min3DHit_;
  double maxChi2_;
  std::vector<reco::TrackBase::TrackQuality> quality_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithm_;
  edm::InputTag bsSrc_;
  container selected_;
};

#endif
