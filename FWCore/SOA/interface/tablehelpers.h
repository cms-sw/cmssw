#ifndef FWCore_SOA_tablehelpers_h
#define FWCore_SOA_tablehelpers_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
//
/**

 Description: classes and functions used by edm::soa::Table

 Usage:
    These are internal details of Table's implementation

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 24 Aug 2017 16:18:18 GMT
//

// system include files
#include <type_traits>
#include <tuple>

// user include files

// forward declarations

namespace edm {
namespace soa {
namespace impl {
  
template<int I>
struct FoundIndex {
  static constexpr int index = I;
};

template <int I, typename T, typename TPL>
struct GetIndex {
  static constexpr int index = std::conditional<std::is_same<T, typename std::tuple_element<I,TPL>::type >::value,
  FoundIndex<I>,
  GetIndex<I+1, T,TPL>>::type::index;
};

}
}
}

#endif
