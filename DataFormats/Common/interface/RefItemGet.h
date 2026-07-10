#ifndef DataFormats_Common_RefItemGet_h
#define DataFormats_Common_RefItemGet_h

/*----------------------------------------------------------------------
  
RefItemGet: Free function to get pointer to a referenced item.


----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreGet.h"

namespace edm {

  namespace refitem {

    template <typename C, typename T, typename F, typename K>
    inline void findRefItem(RefCore const& refCore, C const* container, K const& key) {
      F finder;
      T const* item = finder(*container, key);
      refCore.setProductPtr(item);
    }

    template <typename C, typename T, typename F, typename KEY>
    struct GetRefPtrImpl {
      static T const* getRefPtr_(RefCore const& product, KEY const& key) {
        T const* item = static_cast<T const*>(product.productPtr());
        if (item != nullptr) {
          return item;
        }
        auto prodGetter = product.productGetter();
        if (nullptr == prodGetter) {
          item = static_cast<T const*>(product.productPtr());
          if (item != nullptr) {
            //Another thread updated the value since we checked
            return item;
          }
        }
        C const* prod = edm::template getProductWithCoreFromRef<C>(product, prodGetter);
        /*
        typename C::const_iterator it = prod->begin();
         std::advance(it, item.key());
         T const* p = it.operator->();
        */
        F func;
        item = func(*prod, key);
        product.setProductPtr(item);
        return item;
      }
    };
  }  // namespace refitem

  template <typename C, typename T, typename F, typename KEY>
  inline T const* getRefPtr(RefCore const& product, KEY const& iKey) {
    return refitem::GetRefPtrImpl<C, T, F, KEY>::getRefPtr_(product, iKey);
  }
}  // namespace edm

#endif
