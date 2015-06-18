#ifdef DataFormats_Common_HolderToVectorTrait_h
#ifdef DataFormats_Common_Ref_h

#ifndef DataFormats_Common_HolderToVectorTrait_Ref_specialization_h
#define DataFormats_Common_HolderToVectorTrait_Ref_specialization_h
// -*- C++ -*-
//
// Package:     DataFormats/Common
//
/**

 Description: Specialization of HolderToVectorTrait for edm::Ref

 Usage:
    Only if both DataFormats/Common/interface/HolderToVectorTrait.h and
 DataFormats/Common/interface/Ref.h are include in a file will this code
 be seen.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 20 Nov 2014 21:44:59 GMT
//
#include <memory>

#include "DataFormats/Common/interface/VectorHolder.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"

namespace edm {
  namespace reftobase {

    template <typename T, typename REF>
    struct RefHolderToVector {
      static  std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() {
        typedef RefVector<typename REF::product_type,
        typename REF::value_type,
        typename REF::finder_type> REFV;
        return std::auto_ptr<BaseVectorHolder<T> >(new VectorHolder<T, REFV>);
      }
    };

    template<typename T1, typename C, typename T, typename F>
    struct HolderToVectorTrait<T1, Ref<C, T, F> > {
      typedef RefHolderToVector<T1, Ref<C, T, F> > type;
    };

    template <typename REF>
    struct RefRefHolderToRefVector {
      static std::auto_ptr<RefVectorHolderBase> makeVectorHolder() {
        typedef RefVector<typename REF::product_type,
        typename REF::value_type,
        typename REF::finder_type> REFV;
        return std::auto_ptr<RefVectorHolderBase>(new RefVectorHolder<REFV>);
      }
    };

    template<typename C, typename T, typename F>
    struct RefHolderToRefVectorTrait<Ref<C, T, F> > {
      typedef RefRefHolderToRefVector<Ref<C, T, F> > type;
    };
  }
}

#endif
#endif
#endif
