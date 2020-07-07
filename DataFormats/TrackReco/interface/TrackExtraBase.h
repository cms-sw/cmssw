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
#include "DataFormats/Common/interface/RefItemGet.h"
#include <numeric>

namespace reco {

  class TrackExtraBase {
  public:
    using TrajParams = std::vector<LocalTrajectoryParameters>;
    using Chi2sFive = std::vector<unsigned char>;

    /// default constructor
    TrackExtraBase() : m_firstHit((unsigned int)-1), m_nHits(0) {}

    void setHits(TrackingRecHitRefProd const& prod, unsigned firstH, unsigned int nH) {
      m_hitCollection.pushBackItem(prod.refCore(), true);
      m_firstHit = firstH;
      m_nHits = nH;
    }

    void setTrajParams(TrajParams tmps, Chi2sFive chi2s) {
      m_trajParams = std::move(tmps);
      m_chi2sX5 = std::move(chi2s);
    }

    unsigned int firstRecHit() const { return m_firstHit; }

    /// number of RecHits
    unsigned int recHitsSize() const { return m_nHits; }

    /// accessor to RecHits
    auto recHits() const {
      trackingRecHit_iterator const& begin = recHitsBegin();
      trackingRecHit_iterator end = begin == trackingRecHit_iterator() ? trackingRecHit_iterator() : begin + m_nHits;
      return TrackingRecHitRange(begin, end);
    }

    /// first iterator over RecHits
    trackingRecHit_iterator recHitsBegin() const {
      //Another thread might change the RefCore at the same time.
      // By using a copy we will be safe.
      edm::RefCore hitCollection(m_hitCollection);
      TrackingRecHitCollection const* hits = static_cast<TrackingRecHitCollection const*>(hitCollection.productPtr());
      //if original collection is available, return iterator to it directly
      if (hits != nullptr) {
        return hits->data().begin() + m_firstHit;
      }
      hits =
          edm::tryToGetProductWithCoreFromRef<TrackingRecHitCollection>(hitCollection, hitCollection.productGetter());
      if (hits != nullptr) {
        return hits->data().begin() + m_firstHit;
      }

      //check for thinned collection
      std::vector<edm::WrapperBase const*> prods(m_nHits, nullptr);
      std::vector<unsigned int> keys(m_nHits);
      //fill with sequential integers
      std::iota(keys.begin(), keys.end(), m_firstHit);
      //get thinned hit collections/indices
      hitCollection.productGetter()->getThinnedProducts(hitCollection.id(), prods, keys);
      //check if all hits are available from a single collection and in sequential order
      bool valid = true;
      if (!m_nHits) {
        valid = false;
      } else if (prods.front() == nullptr) {
        valid = false;
      } else if (!std::equal(prods.begin() + 1, prods.end(), prods.begin())) {
        valid = false;
      } else if (!std::is_sorted(keys.begin(), keys.end())) {
        valid = false;
      } else if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
        valid = false;
      } else if ((keys.back() - keys.front()) != m_nHits) {
        valid = false;
      }

      if (valid) {
        hits = static_cast<edm::Wrapper<TrackingRecHitCollection> const*>(prods.front())->product();
        return hits->data().begin() + keys.front();
      }

      return trackingRecHit_iterator();
    }

    /// last iterator over RecHits
    trackingRecHit_iterator recHitsEnd() const {
      trackingRecHit_iterator const& begin = recHitsBegin();
      return begin == trackingRecHit_iterator() ? trackingRecHit_iterator() : begin + m_nHits;
    }

    /// get a ref to i-th recHit
    TrackingRecHitRef recHitRef(unsigned int i) const {
      //Another thread might change the RefCore at the same time.
      // By using a copy we will be safe.
      edm::RefCore hitCollection(m_hitCollection);
      if (hitCollection.productPtr()) {
        TrackingRecHitRef::finder_type finder;
        TrackingRecHitRef::value_type const* item =
            finder(*(static_cast<TrackingRecHitRef::product_type const*>(hitCollection.productPtr())), m_firstHit + i);
        return TrackingRecHitRef(hitCollection.id(), item, m_firstHit + i);
      }
      return TrackingRecHitRef(hitCollection, m_firstHit + i);
    }

    /// get i-th recHit
    TrackingRecHitRef recHit(unsigned int i) const { return recHitRef(i); }

    TrajParams const& trajParams() const { return m_trajParams; }
    Chi2sFive const& chi2sX5() const { return m_chi2sX5; }

  private:
    edm::RefCore m_hitCollection;
    unsigned int m_firstHit;
    unsigned int m_nHits;
    TrajParams m_trajParams;
    Chi2sFive m_chi2sX5;  // chi2 * 5  chopped at 255  (max chi2 is 51)
  };

}  // namespace reco

#endif  // DataFormats_TrackReco_TrackExtraBase_h
