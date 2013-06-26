#ifndef DataFormats_Common_HolderToVectorTrait_h
#define DataFormats_Common_HolderToVectorTrait_h
#include "FWCore/Utilities/interface/EDMException.h"
#include <memory>
//#include <boost/static_assert.hpp>

namespace edm {
  namespace reftobase {
    class RefVectorHolderBase;
    template <typename T> class BaseVectorHolder;

    template <typename T, typename REF>
    struct InvalidHolderToVector {
      static std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() {
	Exception::throwThis(errors::InvalidReference,
	  "InvalidHolderToVector: trying to use RefToBase built with "
	  "an internal type. RefToBase should be built passing an "
	  "object of type edm::Ref<C>. This exception should never "
	  "be thrown if a RefToBase was built from a RefProd<C>.");
        return std::auto_ptr<BaseVectorHolder<T> >();
      }
      static std::auto_ptr<RefVectorHolderBase> makeVectorBaseHolder() {
	Exception::throwThis(errors::InvalidReference,
	  "InvalidHolderToVector: trying to use RefToBase built with "
	  "an internal type. RefToBase should be built passing an "
	  "object of type edm::Ref<C>. This exception should never "
	  "be thrown if a RefToBase was built from a RefProd<C>.");
        return std::auto_ptr<RefVectorHolderBase>();
      }
    };

    template<typename T, typename REF>
    struct HolderToVectorTrait {
      //      BOOST_STATIC_ASSERT(sizeof(REF) == 0); 
      typedef InvalidHolderToVector<T, REF> type;
    };

    template <typename REF>
    struct InvalidRefHolderToRefVector {
      static std::auto_ptr<RefVectorHolderBase> makeVectorHolder() {
	Exception::throwThis(errors::InvalidReference,
	  "InvalidRefHolderToRefVector: trying to use RefToBaseVector built with "
	  "an internal type. RefToBase should be built passing an "
	  "object of type edm::RefVector<C>");
        return std::auto_ptr<RefVectorHolderBase>();
      }
      static std::auto_ptr<RefVectorHolderBase> makeVectorBaseHolder() {
	Exception::throwThis(errors::InvalidReference,
	  "InvalidRefHolderToRefVector: trying to use RefToBaseVector built with "
	  "an internal type. RefToBase should be built passing an "
	  "object of type edm::RefVector<C>");
        return std::auto_ptr<RefVectorHolderBase>();
      }
    };
    
    template<typename REF>
    struct RefHolderToRefVectorTrait {
      //      BOOST_STATIC_ASSERT(sizeof(REF) == 0); 
      typedef InvalidRefHolderToRefVector<REF> type;
    };

  }
}

#endif
