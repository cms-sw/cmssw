#ifndef FWCore_Reflection_TypeInfoList_h
#define FWCore_Reflection_TypeInfoList_h
// -*- C++ -*-
//
// Package:     FWCore/Reflection
// Class  :     TypeInfoList
//
/**\class TypeInfoList TypeInfoList.h "FWCore/Reflection/interface/TypeInfoList.h"

 Description: Converts types in template argument to a container of std::type_infos

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 27 Jul 2022 20:36:03 GMT
//

// system include files
#include <array>
#include <algorithm>

// user include files

// forward declarations

namespace edm {
  template <typename... T>
  struct TypeInfoList;

  template <typename T, typename... REMAINING>
  struct TypeInfoList<T, REMAINING...> {
    static constexpr unsigned int nTypes = TypeInfoList<REMAINING...>::nTypes + 1;
    static constexpr std::array<std::type_info const*, nTypes> list() noexcept {
      auto i = previous_list();
      std::array<std::type_info const*, nTypes> v = {{&typeid(T)}};
      std::copy(i.begin(), i.end(), v.begin() + 1);
      return v;
    };
    template <typename D>
    static constexpr void test_inheritance(D const* iD) {
      test_inheritance_(iD);
      TypeInfoList<REMAINING...>::test_inheritance(iD);
    }

  private:
    static constexpr std::array<std::type_info const*, nTypes - 1> previous_list() noexcept {
      return TypeInfoList<REMAINING...>::list();
    }
    static constexpr void test_inheritance_(T const*) {}
  };

  template <>
  struct TypeInfoList<> {
    static constexpr unsigned int nTypes = 0;
    static constexpr std::array<std::type_info const*, 0> list() noexcept { return {}; };
    template <typename D>
    static constexpr void test_inheritance(D const* iD) {}
  };
}  // namespace edm
#endif
