#ifndef DataFormats_Common_RefItemGet_h
#define DataFormats_Common_RefItemGet_h

/*----------------------------------------------------------------------
  
RefItemGet: Free function to get pointer to a referenced item.

$Id: RefItemGet.h,v 1.3.6.2 2011/02/17 03:09:36 chrjones Exp $

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
  }

  template <typename C, typename T, typename F, typename KEY>
  inline
  T const* getPtr_(RefCore const& product, KEY const& key) {
    return refitem::GetPtrImpl<C, T, F, KEY>::getPtr_(product, key);
  }

  template <typename C, typename T, typename F, typename KEY>
  inline
  T const* getPtr(RefCore const& product, KEY const& iKey, const void*& oPtr) {
    T const* p = static_cast<T const *>(oPtr);
    if(0==p){
      p=refitem::GetPtrImpl<C, T, F, KEY>::getPtr_(product, iKey);
      oPtr = p;
    }
    return p;
  }

}

#endif
