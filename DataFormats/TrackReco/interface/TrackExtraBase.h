#ifndef TrackReco_TrackExtraBase_h
#define TrackReco_TrackExtraBase_h
/** \class reco::TrackExtraBase TrackExtraBase.h DataFormats/TrackReco/interface/TrackExtraBase.h
 *
 * Basic extension of a reconstructed Track. 
 * Contains references to RecHits used in the fit.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: TrackExtraBase.h,v 1.4 2006/04/28 11:23:43 llista Exp $
 *
 */
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

namespace reco {
  class TrackExtraBase {
  public:
    /// default constructor
    TrackExtraBase() { }
    /// add a reference to a RecHit
    void add( const TrackingRecHitRef & r ) { recHits_.push_back( r ); }
    /// first iterator over RecHits
    trackingRecHit_iterator recHitsBegin() const { return recHits_.begin(); }
    /// last iterator over RecHits
    trackingRecHit_iterator recHitsEnd() const { return recHits_.end(); }
    /// number of RecHits
    size_t recHitsSize() const { return recHits_.size(); }
    /// get n-th recHit
    TrackingRecHitRef recHit( size_t i ) const { return recHits_[ i ]; }
    TrackingRecHitRefVector recHits() const {return recHits_;}

  private:
    /// references to RecHits
    TrackingRecHitRefVector recHits_;
  };

}

#endif
