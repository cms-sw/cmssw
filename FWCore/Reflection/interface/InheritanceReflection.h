#ifndef FWCore_Reflection_InheritanceReflection_h
#define FWCore_Reflection_InheritanceReflection_h
// -*- C++ -*-
//
// Package:     FWCore/Reflection
// Class  :     InheritanceReflection
//
/**\class InheritanceReflection InheritanceReflection.h "FWCore/Reflection/interface/InheritanceReflection.h"

 Description: A type-agnostic holder of inheritance information used for runtime reflection

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 27 Jul 2022 20:47:16 GMT
//

// system include files
#include <typeinfo>
#include <cstddef>

// user include files

// forward declarations
namespace edm {
  class InheritanceReflection {
  public:
    class BaseRange {
    public:
      using const_iterator_type = std::type_info const* const*;
      using element_type = std::type_info const* const;

      constexpr BaseRange(const_iterator_type iBegin, const_iterator_type iEnd) : m_begin(iBegin), m_end(iEnd) {}

      const_iterator_type begin() const { return m_begin; }
      const_iterator_type end() const { return m_end; }

      size_t size() const { return m_end - m_begin; }
      bool empty() const { return m_end == m_begin; }

    private:
      const_iterator_type m_begin;
      const_iterator_type m_end;
    };

    constexpr InheritanceReflection(std::type_info const* iType, BaseRange iBases) : m_type(iType), m_bases(iBases) {}

    constexpr std::type_info const& typeInfo() const noexcept { return *m_type; }
    constexpr BaseRange inheritsFrom() const noexcept { return m_bases; }

  private:
    std::type_info const* m_type;
    BaseRange const m_bases;
  };
}  // namespace edm

#endif
