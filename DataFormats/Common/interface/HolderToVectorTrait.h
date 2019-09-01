#ifndef DataFormats_Common_HolderToVectorTrait_h
#define DataFormats_Common_HolderToVectorTrait_h
#include "FWCore/Utilities/interface/EDMException.h"
#include <memory>

namespace edm {
  namespace reftobase {
    class RefVectorHolderBase;
    template <typename T>
    class BaseVectorHolder;

    template <typename T, typename REF>
    struct InvalidHolderToVector {
      static std::unique_ptr<BaseVectorHolder<T> > makeVectorHolder() {
        Exception::throwThis(errors::InvalidReference,
                             "InvalidHolderToVector: trying to use RefToBase built with "
                             "an internal type. RefToBase should be built passing an "
                             "object of type edm::Ref<C>. This exception should never "
                             "be thrown if a RefToBase was built from a RefProd<C>.");
        return std::unique_ptr<BaseVectorHolder<T> >();
      }
    };

    template <typename T, typename REF>
    struct HolderToVectorTrait {
      //      static_assert(sizeof(REF) == 0);
      typedef InvalidHolderToVector<T, REF> type;
    };

    template <typename REF>
    struct InvalidRefHolderToRefVector {
      static std::unique_ptr<RefVectorHolderBase> makeVectorHolder() {
        Exception::throwThis(errors::InvalidReference,
                             "InvalidRefHolderToRefVector: trying to use RefToBaseVector built with "
                             "an internal type. RefToBase should be built passing an "
                             "object of type edm::RefVector<C>");
        return std::unique_ptr<RefVectorHolderBase>();
      }
    };

    template <typename REF>
    struct RefHolderToRefVectorTrait {
      //      static_assert(sizeof(REF) == 0);
      typedef InvalidRefHolderToRefVector<REF> type;
    };

  }  // namespace reftobase
}  // namespace edm

//Handle specialization here
#include "DataFormats/Common/interface/HolderToVectorTrait_Ref_specialization.h"
#include "DataFormats/Common/interface/HolderToVectorTrait_Ptr_specialization.h"
#include "DataFormats/Common/interface/HolderToVectorTrait_RefProd_specialization.h"

#endif
