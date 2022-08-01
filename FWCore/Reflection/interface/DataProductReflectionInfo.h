#ifndef FWCore_Reflection_DataProductReflectionInfo_h
#define FWCore_Reflection_DataProductReflectionInfo_h
// -*- C++ -*-
//
// Package:     FWCore/Reflection
// Class  :     DataProductReflectionInfo
//
/**\class DataProductReflectionInfo DataProductReflectionInfo.h "FWCore/Reflection/interface/DataProductReflectionInfo.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 27 Jul 2022 20:49:53 GMT
//

// system include files

// user include files
#include "FWCore/Reflection/interface/InheritanceReflection.h"
#include "FWCore/Reflection/interface/StaticDataProductReflection.h"

// forward declarations
namespace edm {
  struct DataProductReflectionInfo {
    using BaseRange = InheritanceReflection::BaseRange;

    constexpr DataProductReflectionInfo(InheritanceReflection iType, std::type_info const* iElementType)
        : m_type(iType), m_elementType(iElementType) {}

    constexpr std::type_info const& typeInfo() const noexcept { return m_type.typeInfo(); }
    constexpr BaseRange inheritsFrom() const noexcept { return m_type.inheritsFrom(); }

    constexpr bool isContainer() const noexcept { return static_cast<bool>(m_elementType); }
    constexpr std::type_info const& elementType() const noexcept {
      if (m_elementType) {
        return *m_elementType;
      }
      return m_type.typeInfo();
    }

  private:
    InheritanceReflection const m_type;
    std::type_info const* const m_elementType;
  };

  template <typename T>
  //can't make this constexpr until C++23
  const DataProductReflectionInfo makeDataProductReflectionInfo() {
    using Reflectn = StaticDataProductReflection<T>;
    static auto const s_bases = Reflectn::inherits_from();
    InheritanceReflection reflection(&typeid(T),
                                     InheritanceReflection::BaseRange(s_bases.data(), s_bases.data() + s_bases.size()));
    if constexpr (Reflectn::is_container) {
      return DataProductReflectionInfo{reflection, &typeid(typename Reflectn::element_type)};
    }
    return DataProductReflectionInfo{reflection, nullptr};
  }

}  // namespace edm
#endif
