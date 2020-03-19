#ifndef DataFormats_Common_GetProduct_h
#define DataFormats_Common_GetProduct_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     GetProduct
//
/**\class GetProduct GetProduct.h DataFormats/Common/interface/GetProduct.h

 Description: Controls how edm::View and edm::Ptr interact with containers

 Usage:
    Override this class in order to specialize edm::View or edm::Ptr's interaction with a container

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 20 10:20:20 EDT 2007
//

// system include files

#include <memory>
#include <vector>

// user include files

// forward declarations

namespace edm {
  namespace detail {
    template <typename COLLECTION>
    struct GetProduct {
      typedef typename COLLECTION::value_type element_type;
      typedef typename COLLECTION::const_iterator iter;
      static const element_type* address(const iter& i) { return &*i; }
    };

    // Specialize for vector<unique_ptr<T>>>
    template <typename T, typename D, typename A>
    struct GetProduct<std::vector<std::unique_ptr<T, D>, A> > {
      using element_type = T;
      using iter = typename std::vector<std::unique_ptr<T, D>, A>::const_iterator;
      static const element_type* address(const iter& i) { return i->get(); }
    };
  }  // namespace detail
}  // namespace edm

#endif
