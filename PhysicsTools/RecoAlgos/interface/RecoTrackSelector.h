#ifndef RecoSelectors_RecoTrackSelector_h
#define RecoSelectors_RecoTrackSelector_h
/* \class RecoTrackSelector
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2008/02/13 10:42:23 $
 *  $Revision: 1.4 $
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class RecoTrackSelector {
 public:
  typedef reco::TrackCollection collection;
  typedef std::vector<const reco::Track *> container;
  typedef container::const_iterator const_iterator;

  /// Constructors
  RecoTrackSelector() {}
  RecoTrackSelector ( const edm::ParameterSet & cfg ) {
    RecoTrackSelector(cfg.getParameter<double>("ptMin"),
		      cfg.getParameter<double>("minRapidity"),
		      cfg.getParameter<double>("maxRapidity"),
		      cfg.getParameter<double>("tip"),
		      cfg.getParameter<double>("lip"),
		      cfg.getParameter<int>("minHit"),
		      cfg.getParameter<double>("maxChi2"));
  }

  RecoTrackSelector ( double ptMin, double minRapidity, double maxRapidity,
		      double tip, double lip, int minHit, double maxChi2 ) :
    ptMin_( ptMin ), minRapidity_( minRapidity ), maxRapidity_( maxRapidity ),
    tip_( tip ), lip_( lip ), minHit_( minHit ), maxChi2_( maxChi2 ) { }

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  
  void select( const edm::Handle<collection>& c, const edm::Event & event, const edm::EventSetup&) {
    selected_.clear();
    edm::Handle<reco::BeamSpot> beamSpot;
    event.getByLabel("offlineBeamSpot",beamSpot); 
    for( reco::TrackCollection::const_iterator trk = c->begin(); 
         trk != c->end(); ++ trk )
      if ( operator()(*trk,beamSpot.product()) ) selected_.push_back( & * trk );
  }

  /// Operator() performs the selection: e.g. if (recoTrackSelector(track)) {...}
  bool operator()( const reco::Track & t, const reco::BeamSpot* bs) {
    return
      (t.numberOfValidHits() >= minHit_ &&
       fabs(t.pt()) >= ptMin_ &&
       t.eta() >= minRapidity_ && t.eta() <= maxRapidity_ &&
       fabs(t.dxy(bs->position())) <= tip_ &&
       fabs(t.dz()) <= lip_  &&
       t.normalizedChi2()<=maxChi2_);
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
  container selected_;
};

#endif
