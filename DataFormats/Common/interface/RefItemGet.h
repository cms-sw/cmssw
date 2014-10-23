#ifndef DataFormats_Common_RefItemGet_h
#define DataFormats_Common_RefItemGet_h

/*----------------------------------------------------------------------
  
RefItemGet: Free function to get pointer to a referenced item.


----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreGet.h"

namespace edm {
  
  namespace refitem {
    template< typename C, typename T, typename F, typename KEY>
    struct GetPtrImpl {
      static T const* getPtr_(RefCore const& product, KEY const& key) {
        C const* prod = edm::template getProduct<C>(product);
        /*
        typename C::const_iterator it = prod->begin();
         std::advance(it, item.key());
         T const* p = it.operator->();
        */
        F func;
        T const* p = func(*prod, key);
        return p;
      }
    };

    template< typename C, typename T, typename F>
    struct GetPtrImpl<C, T, F, unsigned int> {
      static T const* getPtr_(RefCore const& product, unsigned int key) {
        C const* prod = edm::template tryToGetProduct<C>(product);
        if(prod != nullptr) {
          F func;
          T const* p = func(*prod, key);
          return p;
        }
        unsigned int thinnedKey = key;
        prod = edm::template getThinnedProduct<C>(product, thinnedKey);
        F func;
        T const* p = func(*prod, thinnedKey);
        return p;
      }
    };
  }

  template <typename C, typename T, typename F, typename KEY>
  inline
  T const* getPtr_(RefCore const& product, KEY const& key) {
    return refitem::GetPtrImpl<C, T, F, KEY>::getPtr_(product, key);
  }

  template <typename C, typename T, typename F, typename KEY>
  inline
  T const* getPtr(RefCore const& product, KEY const& iKey) {
    T const* p=refitem::GetPtrImpl<C, T, F, KEY>::getPtr_(product, iKey);
    return p;
  }

  namespace refitem {
    template< typename C, typename KEY>
    struct IsThinnedAvailableImpl {
      static bool isThinnedAvailable_(RefCore const& product, KEY const& key) {
        return false;
      }
    };

    template< typename C >
    struct IsThinnedAvailableImpl<C, unsigned int> {
      static bool isThinnedAvailable_(RefCore const& ref, unsigned int key) {
        if(ref.productPtr() != nullptr) {
          return true;
        }
        if (ref.isTransient()) {
          return false;
        }
        return ref.isThinnedAvailable(key);
      }
    };
  }

  template <typename C, typename KEY>
  inline
  bool isThinnedAvailable(RefCore const& product, KEY const& iKey) {
    return refitem::IsThinnedAvailableImpl<C, KEY>::isThinnedAvailable_(product, iKey);
  }
}

#endif
