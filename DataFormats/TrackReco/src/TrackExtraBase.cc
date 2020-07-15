#include "DataFormats/TrackReco/interface/TrackExtraBase.h"

using namespace reco;

trackingRecHit_iterator TrackExtraBase::recHitsBegin() const {
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
