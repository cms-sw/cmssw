#ifndef DataFormats_Common_RefCoreGet_h
#define DataFormats_Common_RefCoreGet_h

/*----------------------------------------------------------------------
  
RefCoreGet: Free function to get the pointer to a referenced product.

$Id: RefCoreGet.h,v 1.6 2008/09/05 17:53:23 chrjones Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace edm {

  namespace refcore {
    template <typename T>
    inline
    T const* 
    getProductPtr_(RefCore const& ref) {
      //if (isNull()) throwInvalidReference();
      assert (!ref.isTransient());
      EDProduct const* product = ref.getProductPtr(typeid(T).name());
      Wrapper<T> const* wrapper = dynamic_cast<Wrapper<T> const*>(product);

      if (wrapper == 0) { 
        ref.wrongTypeException(typeid(T).name(), typeid(*product).name());
      }
      ref.setProductPtr(wrapper->product());
      return wrapper->product();
    }
  }

  template <typename T>
  inline
  T const*
  getProduct(RefCore const & ref) {
    T const* p = static_cast<T const *>(ref.productPtr());
    if (p != 0) return p;
    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T).name());
    }
    return refcore::getProductPtr_<T>(ref);
  }
}
#endif
