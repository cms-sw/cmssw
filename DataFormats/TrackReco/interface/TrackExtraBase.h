#ifndef DataFormats_TrackReco_TrackExtraBase_h
#define DataFormats_TrackReco_TrackExtraBase_h

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
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

namespace reco
{

class TrackExtraBase
{
public:
    using TrajParams = std::vector<LocalTrajectoryParameters>;
    using Chi2sFive = std::vector<unsigned char>;

    /// default constructor
    TrackExtraBase() : m_firstHit((unsigned int) -1), m_nHits(0) { }

    void setHits(TrackingRecHitRefProd const & prod, unsigned firstH, unsigned int nH) {
        m_hitCollection.pushBackItem(prod.refCore(), true);
        m_firstHit = firstH;
        m_nHits = nH;
    }

    void setTrajParams(TrajParams tmps, Chi2sFive chi2s) {
      m_trajParams = std::move(tmps);
      m_chi2sX5 = std::move(chi2s);
    }

    unsigned int firstRecHit() const {
      return m_firstHit;
    }

    /// number of RecHits
    unsigned int recHitsSize() const {
        return m_nHits;
    }

    /// accessor to RecHits
    auto recHits() const { return TrackingRecHitRange(recHitsBegin(), recHitsEnd()); }

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
      //Another thread might change the RefCore at the same time.
      // By using a copy we will be safe.
      edm::RefCore hitCollection(m_hitCollection);
      if(hitCollection.productPtr()) {
        TrackingRecHitRef::finder_type finder;
        TrackingRecHitRef::value_type const* item = finder(*(static_cast<TrackingRecHitRef::product_type const*>(hitCollection.productPtr())), m_firstHit+i);
        return TrackingRecHitRef(hitCollection.id(), item, m_firstHit+i);
      }
      return TrackingRecHitRef(hitCollection, m_firstHit+i);
    }

    /// get i-th recHit
    TrackingRecHitRef recHit(unsigned int i) const {
        return recHitRef(i);
    }

    TrackingRecHitCollection const & recHitsProduct() const {
      return *edm::getProduct<TrackingRecHitCollection>(m_hitCollection);
    }

    TrajParams const & trajParams() const  {return m_trajParams;}
    Chi2sFive const & chi2sX5() const { return m_chi2sX5;}

private:
    edm::RefCore m_hitCollection;
    unsigned int m_firstHit;
    unsigned int m_nHits;
    TrajParams m_trajParams;
    Chi2sFive m_chi2sX5;  // chi2 * 5  chopped at 255  (max chi2 is 51)
};

} // namespace reco

#endif // DataFormats_TrackReco_TrackExtraBase_h
