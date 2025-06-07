#ifndef FWCore_Reflection_StaticDataProductReflection_h
#define FWCore_Reflection_StaticDataProductReflection_h
// -*- C++ -*-
//
// Package:     FWCore/Reflection
// Class  :     StaticDataProductReflection
//
/**\class StaticDataProductReflection StaticDataProductReflection.h "FWCore/Reflection/interface/StaticDataProductReflection.h"

 Description: template class describing compile time infor about a data product

 Usage:
    Developers must specialize a StaticDataProductReflection for each data product type. This is done by inheriting from
    StaticDataProductReflectionBase and declaring all inherited classes in the template arguments

    template<> class StaticDataProductReflection<Foo> : public StaticDataProductReflectionBase<Foo, Bar> {};

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 27 Jul 2022 20:39:27 GMT
//

// system include files
#include <vector>
#include <array>
#include <typeinfo>

// user include files
#include "FWCore/Reflection/interface/TypeInfoList.h"

// forward declarations

namespace edm {
  template <typename T, typename... BASES>
  struct StaticDataProductReflectionBase {
    static constexpr bool is_container = false;
    using element_type = T;
    using Inheritance = TypeInfoList<BASES...>;

    static constexpr unsigned int nBases = Inheritance::nTypes;
    static constexpr std::array<std::type_info const*, nBases> inherits_from() noexcept {
      Inheritance::test_inheritance(static_cast<T const*>(nullptr));
      return Inheritance::list();
    };
  };

  template <typename T>
  struct StaticDataProductReflectionBase<T> {
    static constexpr bool is_container = false;
    using element_type = T;

    static constexpr unsigned int nBases = 0;
    static constexpr std::array<std::type_info const*, 0> inherits_from() noexcept { return {}; };

    static constexpr std::array<std::type_info const*, 1> class_and_inherits_from() noexcept { return {{&typeid(T)}}; };
  };

  template <typename T, typename... U>
  struct StaticDataProductReflectionBase<std::vector<T, U...>> {
    static constexpr bool is_container = true;

    using element_type = typename std::vector<T, U...>::value_type;

    static constexpr unsigned int nBases = 0;
    static constexpr std::array<std::type_info const*, 0> inherits_from() noexcept { return {}; };
  };

  template <typename T>
  struct StaticDataProductReflection;

}  // namespace edm
#endif
