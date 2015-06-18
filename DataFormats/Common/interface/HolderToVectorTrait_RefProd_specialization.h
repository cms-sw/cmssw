#ifdef DataFormats_Common_HolderToVectorTrait_h
#ifdef DataFormats_Common_RefProd_h

#ifndef DataFormats_Common_HolderToVectorTrait_RefProd_specialization_h
#define DataFormats_Common_HolderToVectorTrait_RefProd_specialization_h
// -*- C++ -*-
//
// Package:     DataFormats/Common
//
/**

 Description: Specialization of HolderToVectorTrait for edm::RefProd

 Usage:
    Only if both DataFormats/Common/interface/HolderToVectorTrait.h and
 DataFormats/Common/interface/RefProd.h are include in a file will this code
 be seen.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 20 Nov 2014 21:44:59 GMT
//
#include <memory>

namespace edm {
  namespace reftobase {

    template<typename T>
    struct RefProdHolderToVector {
      static  std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() {
        Exception::throwThis(errors::InvalidReference, "attempting to make a BaseVectorHolder<T> from a RefProd<C>.\n");
        return std::auto_ptr<BaseVectorHolder<T> >();
      }
    };

    template<typename C, typename T>
    struct HolderToVectorTrait<T, RefProd<C> > {
      typedef RefProdHolderToVector<T> type;
    };

    struct RefProdRefHolderToRefVector {
      static std::auto_ptr<RefVectorHolderBase> makeVectorHolder() {
        Exception::throwThis(errors::InvalidReference, "attempting to make a BaseVectorHolder<T> from a RefProd<C>.\n");
        return std::auto_ptr<RefVectorHolderBase>();
      }
    };

    template<typename C>
    struct RefHolderToRefVectorTrait<RefProd<C> > {
      typedef RefProdRefHolderToRefVector type;
    };
  }
}

#endif
#endif
#endif
