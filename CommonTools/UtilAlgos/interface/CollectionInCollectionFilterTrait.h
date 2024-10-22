#ifndef UtilAlgos_CollectionInCollectionFilterTrait_h
#define UtilAlgos_CollectionInCollectionFilterTrait_h
/* \class CollectionInCollectionFilterTrait<C, S, N>
 *
 * \author Marco Musich
 *
 */
#include "CommonTools/UtilAlgos/interface/AnySelector.h"
#include "CommonTools/UtilAlgos/interface/MinNumberSelector.h"

namespace helper {

  template <typename C, typename N>
  struct TotalSizeFilter {
    template <typename S>
    static bool filter(const C& source, const S&, const N& sizeSelect) {
      size_t n = 0;
      // this assumes the input collection is a DetSetVector
      // loop on the list of DetSet-s
      for (const auto& i : source) {
        // loop on the content of the DetSet
        for (const auto& j : i) {
          n += j.size();
        }
      }
      if (sizeSelect(n))
        return true;
      else
        return false;
    }
  };

  template <typename C, typename S, typename N>
  struct CollectionInCollectionFilterTrait {
    typedef TotalSizeFilter<C, N> type;
  };

}  // namespace helper

#endif
