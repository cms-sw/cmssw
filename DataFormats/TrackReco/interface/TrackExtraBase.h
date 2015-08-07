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
    TrackExtraBase() : m_firstHit(-1), m_nHits(0) { }

    void setHits(TrackingRecHitRefProd const & prod, unsigned firstH, unsigned int nH) {
        m_hitCollection.pushBackItem(prod.refCore(),true);
        m_firstHit =firstH;  m_nHits=nH;
    }

    unsigned int firstRecHit() const {
      return m_firstHit;
    }

    /// number of RecHits
    unsigned int recHitsSize() const {
        return m_nHits;
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
   TrackingRecHitRef recHitRef(unsigned int i) const {                                                               
      return TrackingRecHitRef(m_hitCollection,m_firstHit+i);
    }

    /// get i-th recHit
    TrackingRecHitRef recHit(unsigned int i) const {
        return recHitRef(i);
    }

    TrackingRecHitCollection const & recHitsProduct() const {
      return *edm::getProduct<TrackingRecHitCollection>(m_hitCollection);

    }

private:

    edm::RefCore m_hitCollection;
    unsigned int m_firstHit;
    unsigned int m_nHits;

};

}// namespace reco

#endif

