#ifndef DataFormats_Common_RefToPtr_h
#define DataFormats_Common_RefToPtr_h

/*----------------------------------------------------------------------
  
Ref: A function template for conversion from RefToBase to Ptr

----------------------------------------------------------------------*/
/*
    ----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace edm {
  template <typename T>
  Ptr<T> refToBaseToPtr(RefToBase<T> const& ref) {
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
    // If this is called in an iorule outside the framework, we cannot call
    // ref.get() but since the Ptr will be able to get it later anyway, we can
    // fill with a nullptr for now
    return Ptr<T>(ref.id(), static_cast<const T*>(nullptr), ref.key());
  }
}  // namespace edm
#endif
