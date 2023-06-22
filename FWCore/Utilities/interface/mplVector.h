#ifndef FWCore_Utilities_mplVector_h
#define FWCore_Utilities_mplVector_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     edm::mpl::Vector
//
/**\class edm::mpl::Vector mplVector.h "FWCore/Utilities/interface/mplVector.h"

 Description: A meta-programming container of types

 Usage:
  The collection of types are specified in the template parameters.
  \code
    using Types = edm::mpl::Vector<A,B,C>;
  \endcode

  A check on if a type is held by the container can be done via the call to contains
  \code
  static_assert(edm::mpl::Vector<A,B,C>::contains<A>());
  \endcode
 
  It is possible to move through the sequence of types using the helper edm::mpl::Pop struct
  \code
     using edm::mpl;
     //get the first item
     static_assert(std::is_same_v<Pop<Vector<A,B>>::Item, A>);
 
     //get a container holding what remains
     static_assert(std::is_same_v<Vector<B>, Pop<Vector<A,B>::Remaining>);
 
     //check if more there
     static_assert(not Pop<Vector<B>>::empty);
     static_assert(Pop<Vector<>>::empty);
  \endcode
*/
//
// Original Author:  Chris Jones
//         Created:  Tues, 21 Jul 2020 14:29:51 GMT
//

// system include files
#include <type_traits>

// user include files

// forward declarations

namespace edm {
  namespace mpl {
    template <typename... T>
    class Vector {
    public:
      ///Returns true if the type U is within the collection
      template <typename U>
      static constexpr bool contains() {
        return (std::is_same_v<U, T> || ...);
      }
    };

    template <typename T>
    struct Pop;

    template <typename F, typename... T>
    struct Pop<Vector<F, T...>> {
      constexpr static bool empty = false;
      using Item = F;
      using Remaining = Vector<T...>;
    };

    template <>
    struct Pop<Vector<>> {
      constexpr static bool empty = true;
      using Item = void;
      using Remaining = Vector<>;
    };

    template <typename T, typename U>
    struct Push;

    template <typename T, typename... U>
    struct Push<T, Vector<U...>> {
      constexpr static bool empty = false;
      using Result = Vector<T, U...>;
    };
  }  // namespace mpl
}  // namespace edm

#endif
