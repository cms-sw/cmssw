#ifndef RecoSelectors_RecoTrackSelector_h
#define RecoSelectors_RecoTrackSelector_h
/* \class RecoTrackSelector
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2008/07/24 09:12:55 $
 *  $Revision: 1.12 $
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

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
    maxChi2_(cfg.getParameter<double>("maxChi2")),
    quality_(cfg.getParameter<std::vector<std::string> >("quality")),
    algorithm_(cfg.getParameter<std::vector<std::string> >("algorithm")),
    bsSrc_(cfg.getParameter<edm::InputTag>("beamSpot")) { }

  RecoTrackSelector ( double ptMin, double minRapidity, double maxRapidity,
		      double tip, double lip, int minHit, double maxChi2, 
		      std::vector<std::string> quality , std::vector<std::string> algorithm ) :
    ptMin_( ptMin ), minRapidity_( minRapidity ), maxRapidity_( maxRapidity ),
    tip_( tip ), lip_( lip ), minHit_( minHit ), maxChi2_( maxChi2 ),
    quality_(quality),algorithm_(algorithm) { }

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
	if (t.quality(t.qualityByName(quality_[i]))){
	  quality_ok = true;
	  break;	  
	}
      }
    }
    bool algo_ok = true;
    if (algorithm_.size()!=0) {
      if (std::find(algorithm_.begin(),algorithm_.end(),t.algoName())==algorithm_.end()) algo_ok = false;
    }
    return
      (t.hitPattern().trackerLayersWithMeasurement() >= minHit_ &&
       fabs(t.pt()) >= ptMin_ &&
       t.eta() >= minRapidity_ && t.eta() <= maxRapidity_ &&
       fabs(t.dxy(bs->position())) <= tip_ &&
       fabs(t.dsz(bs->position())) <= lip_  &&
       t.normalizedChi2()<=maxChi2_ &&
       quality_ok &&
       algo_ok);
  }

  size_t size() const { return selected_.size(); }
  
 private:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  int    minHit_;
  double maxChi2_;
  std::vector<std::string> quality_;
  std::vector<std::string> algorithm_;
  edm::InputTag bsSrc_;
  container selected_;
};

#endif
