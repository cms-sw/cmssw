#ifndef DataFormats_Common_RefToPtr_h
#define DataFormats_Common_RefToPtr_h

/*----------------------------------------------------------------------
  
Ref: A function template for conversion from Ref to Ptr

----------------------------------------------------------------------*/
/*
    ----------------------------------------------------------------------*/ 

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefTraits.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace edm {
  template <typename C>
  Ptr<typename C::value_type> refToPtr(
    Ref<C, typename C::value_type, refhelper::FindUsingAdvance<C, typename C::value_type> > const& ref) {
    typedef typename C::value_type T;
    if (ref.isTransient()) {
      return Ptr<T>(ref.product(), ref.key());
    } else if (not ref.hasProductCache()) {
      return Ptr<T>(ref.id(), ref.key(), ref.productGetter());
    }
    return Ptr<T>(ref.id(), ref.get(), ref.key());
  }
}
#endif
