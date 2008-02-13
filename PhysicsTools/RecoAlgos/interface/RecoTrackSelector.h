#ifndef RecoSelectors_RecoTrackSelector_h
#define RecoSelectors_RecoTrackSelector_h
/* \class RecoTrackSelector
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2007/12/13 15:44:25 $
 *  $Revision: 1.2 $
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
<<<<<<< RecoTrackSelector.h
    tip_( tip ), lip_( lip ), minHit_( minHit ), maxChi2_( maxChi2 ) { }

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  
  void select( const edm::Handle<collection>& c, const edm::Event & event) {
    selected_.clear();
    edm::Handle<reco::BeamSpot> beamSpot;
    event.getByLabel("offlineBeamSpot",beamSpot); 
    for( reco::TrackCollection::const_iterator trk = c->begin(); 
         trk != c->end(); ++ trk )
      if ( operator()(*trk,beamSpot.product()) ) selected_.push_back( & * trk );
  }
=======
    tip_( tip ), lip_( lip ), minHit_( minHit ), maxChi2_( maxChi2 ) { }
>>>>>>> 1.2

  /// Operator() performs the selection: e.g. if (recoTrackSelector(track)) {...}
  bool operator()( const reco::Track & t, const reco::BeamSpot* bs) {
    return
      (t.numberOfValidHits() >= minHit_ &&
       fabs(t.pt()) >= ptMin_ &&
       t.eta() >= minRapidity_ && t.eta() <= maxRapidity_ &&
<<<<<<< RecoTrackSelector.h
       fabs(t.dxy(bs->position())) <= tip_ &&
       fabs(t.dz()) <= lip_ );
=======
       fabs(t.d0()) <= tip_ &&
       fabs(t.dz()) <= lip_ &&
       t.normalizedChi2()<=maxChi2_);
>>>>>>> 1.2
  }

  size_t size() const { return selected_.size(); }
  
 private:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  int    minHit_;
<<<<<<< RecoTrackSelector.h
  double maxChi2_;
  container selected_;
=======
  double maxChi2_;

>>>>>>> 1.2
};

<<<<<<< RecoTrackSelector.h
=======
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<RecoTrackSelector> {
      static RecoTrackSelector make( const edm::ParameterSet & cfg ) {
	return RecoTrackSelector(    
          cfg.getParameter<double>( "ptMin" ),
	  cfg.getParameter<double>( "minRapidity" ),
	  cfg.getParameter<double>( "maxRapidity" ),
	  cfg.getParameter<double>( "tip" ),
	  cfg.getParameter<double>( "lip" ),
	  cfg.getParameter<int>( "minHit" ), 
	  cfg.getParameter<double>( "maxChi2" ) ); 
      }
    };
    
  }
}

>>>>>>> 1.2
#endif
