#ifndef TrackReco_TrackExtraBase_h
#define TrackReco_TrackExtraBase_h
/** \class reco::TrackExtraBase
 *
 * Basic extension of a reconstructed Track. 
 * Contains references to RecHits used in the fit.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: TrackExtraBase.h,v 1.6 2006/03/01 14:32:54 llista Exp $
 *
 */
#include "DataFormats/TrackReco/interface/RecHitFwd.h"

namespace reco {
  class TrackExtraBase {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// default constructor
    TrackExtraBase() { }
    /// add a reference to a RecHit
    void add( const RecHitRef & r ) { recHits_.push_back( r ); }
    /// first iterator over RecHits
    recHit_iterator recHitsBegin() const { return recHits_.begin(); }
    /// last iterator over RecHits
    recHit_iterator recHitsEnd() const { return recHits_.end(); }
    /// number of RecHits
    size_t recHitsSize() const { return recHits_.size(); }

  private:
    /// references to RecHits
    RecHitRefs recHits_;
  };

}

#endif
