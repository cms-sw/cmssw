#ifndef DataFormats_Common_RefCoreGet_h
#define DataFormats_Common_RefCoreGet_h

/*----------------------------------------------------------------------

RefCoreGet: Free function to get the pointer to a referenced product.

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
      EDProduct const* product = ref.getProductPtr(typeid(T));
      Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(product);
      if (wrapper == nullptr) { 	 
        ref.wrongTypeException(typeid(T), typeid(*product)); 	 
      }
      ref.setProductPtr(wrapper->product());
      return wrapper->product();
    }
  }

  template <typename T>
  inline
  T const*
  getProduct(RefCore const& ref) {
    T const* p = static_cast<T const*>(ref.productPtr());
    if (p != 0) return p;
    if (ref.isTransient()) {
      ref.nullPointerForTransientException(typeid(T));
    }
    return refcore::getProductPtr_<T>(ref);
  }
}
#endif
