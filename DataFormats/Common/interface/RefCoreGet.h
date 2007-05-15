#ifndef DataFormats_Common_RefCoreGet_h
#define DataFormats_Common_RefCoreGet_h

/*----------------------------------------------------------------------
  
RefCoreGet: Free function to get the pointer to a referenced product.

$Id: RefCoreGet.h,v 1.11 2007/03/29 22:57:25 wmtan Exp $

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
      ref.checkDereferenceability();
      //if (isNull()) throwInvalidReference();

      EDProduct const* product = ref.productGetter()->getIt(ref.id());      
      Wrapper<T> const* wrapper = dynamic_cast<Wrapper<T> const*>(product);
      if (wrapper == 0) { ref.throwWrongReferenceType(typeid(product).name(), typeid(T).name()); }
      ref.setProductPtr(wrapper->product());
      return wrapper->product();
    }
  }

  template <typename T>
  inline
  T const*
  getProduct(RefCore const & ref) {
    T const* p = static_cast<T const *>(ref.productPtr());
    return (p != 0) ? p : refcore::getProductPtr_<T>(ref);
  }
}
#endif
