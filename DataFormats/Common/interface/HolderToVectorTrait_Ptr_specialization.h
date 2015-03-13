#ifdef DataFormats_Common_HolderToVectorTrait_h
#ifdef DataFormats_Common_Ptr_h

#ifndef DataFormats_Common_HolderToVectorTrait_Ptr_specialization_h
#define DataFormats_Common_HolderToVectorTrait_Ptr_specialization_h
// -*- C++ -*-
//
// Package:     DataFormats/Common
//
/**

 Description: Specialization of HolderToVectorTrait for edm::Ptr

 Usage:
    Only if both DataFormats/Common/interface/HolderToVectorTrait.h and
 DataFormats/Common/interface/Ptr.h are include in a file will this code
 be seen.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 20 Nov 2014 21:44:59 GMT
//
#include <memory>

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/VectorHolder.h"

namespace edm {
  template <typename T> class PtrVector;
  namespace reftobase {

    template <typename T, typename U>
    struct PtrHolderToVector {
      static  std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() {
        return std::auto_ptr<BaseVectorHolder<T> >(new VectorHolder<T, edm::PtrVector<U> >);
      }
    };

    template<typename T, typename U>
    struct HolderToVectorTrait<T, Ptr<U> > {
      typedef PtrHolderToVector<T, U > type;
    };

    template <typename T>
    struct PtrRefHolderToRefVector {
      static std::auto_ptr<RefVectorHolderBase> makeVectorHolder() {
        return std::auto_ptr<RefVectorHolderBase>(new RefVectorHolder<edm::PtrVector<T> >);
      }
    };

    template<typename T>
    struct RefHolderToRefVectorTrait<Ptr<T> > {
      typedef PtrRefHolderToRefVector<T> type;
    };
  }
}

#endif
#endif
#endif
