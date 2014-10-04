#ifndef TrackReco_TrackExtraBase_h
#define TrackReco_TrackExtraBase_h
/** \class reco::TrackExtraBase TrackExtraBase.h DataFormats/TrackReco/interface/TrackExtraBase.h
 *
 * Basic extension of a reconstructed Track.
 * Contains references to the hits assigned to the track.
 *
 * If you access the hits, check if they are valid or not. (Invalid hits are dummy hits
 * created in layers crossed by the track, where no physical hit was found).
 *
 * \author Luca Lista, INFN
 *
 *
 */

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

namespace reco
{

class TrackExtraBase
{

public:
    /// default constructor
    TrackExtraBase() { }

    /// add a reference to a RecHit
    void add(const TrackingRecHitRef &r) {
        recHits_.push_back(r);
    }


    unsigned int firstRecHit() const {
      return recHits_.begin().key();
    }

    /// number of RecHits
    size_t recHitsSize() const {
        return recHits_.size();
    }


    /// first iterator over RecHits
    trackingRecHit_iterator recHitsBegin() const {
        return recHitsProduct().data().begin()+firstRecHit();
    }

    /// last iterator over RecHits
    trackingRecHit_iterator recHitsEnd() const {
        return recHitsBegin()+recHitsSize();
    }

    /// get a ref to i-th recHit
    TrackingRecHitRef recHitRef(size_t i) const {                                                               
        return recHits_[i];
    }

    /// get i-th recHit
    TrackingRecHitRef recHit(size_t i) const {
        return recHits_[i];
    }

//    TrackingRecHitRefVector const & recHits() const {
//        return recHits_;
//    }

    TrackingRecHitCollection const & recHitsProduct() const {
      return *recHits_.product();

    }

private:
    /// references to the hit assigned to the track.
    TrackingRecHitRefVector recHits_;

};

}// namespace reco

#endif

