#ifndef DataFormats_Common_RefItemGet_h
#define DataFormats_Common_RefItemGet_h

/*----------------------------------------------------------------------
  
RefItemGet: Free function to get pointer to a referenced item.

$Id: RefItemGet.h,v 1.1 2007/05/15 17:10:24 wmtan Exp $

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/RefCoreGet.h"

namespace edm {
  
  namespace refitem {
    template< typename C, typename T, typename F, typename KEY>
    struct GetPtrImpl {
      static T const* getPtr_(RefCore const& product, RefItem<KEY> const& item) {
        C const* prod = edm::template getProduct<C>(product);
        /*
        typename C::const_iterator it = prod->begin();
         std::advance(it, item.key());
         T const* p = it.operator->();
        */
        F func;
        T const* p = func(*prod, item.key());
        return p;
      }
    };
  }

  template <typename C, typename T, typename F, typename KEY>
  inline
  T const* getPtr_(RefCore const& product, RefItem<KEY> const& item) {
    return refitem::GetPtrImpl<C, T, F, KEY>::getPtr_(product, item);
  }

  template <typename C, typename T, typename F, typename KEY>
  inline
  T const* getPtr(RefCore const& product, RefItem<KEY> const& item) {
    T const* p = static_cast<T const *>(item.ptr());
    return (p != 0 ? p : getPtr_<C, T, F>(product, item));
  }
}

#endif
