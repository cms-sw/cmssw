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
    if (ref.isNull()) {
      return Ptr<T>();
    }
    if (ref.isTransient()) {
      return Ptr<T>(ref.get(), ref.key());
    } else {
      //Another thread could change this value so get only once
      EDProductGetter const* getter = ref.productGetter();
      if (getter) {
        return Ptr<T>(ref.id(), ref.key(), getter);
      }
    }
    return Ptr<T>(ref.id(), ref.get(), ref.key());
  }
}
#endif
