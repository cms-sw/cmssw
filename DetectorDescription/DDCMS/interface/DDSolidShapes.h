#ifndef DETECTOR_DESCRIPTION_DDCMS_DD_SOLID_SHAPES_H
#define DETECTOR_DESCRIPTION_DDCMS_DD_SOLID_SHAPES_H

#include <algorithm>
#include <array>
#include <iterator>
#include <string>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"

using LegacySolidShape = DDSolidShape;

namespace cms {
  namespace dd {
    template <class T>
    struct NameValuePair {
      using value_type = T;
      const T value;
      const char* const name;
    };

    template <class T, class U>
    struct ValuePair {
      using value_type = T;
      using name_type = U;
      const T value;
      const U name;
    };

    template <class Mapping, class V>
    std::string name(Mapping a, V value) {
      auto pos = std::find_if(
          std::begin(a), std::end(a), [&value](const typename Mapping::value_type& t) { return (t.value == value); });
      if (pos != std::end(a)) {
        return pos->name;
      }

      return std::begin(a)->name;
    }

    template <class Mapping>
    typename Mapping::value_type::value_type value(Mapping a, const std::string& name) {
      auto pos = std::find_if(
          std::begin(a), std::end(a), [&name](const typename Mapping::value_type& t) { return (t.name == name); });
      if (pos != std::end(a)) {
        return pos->value;
      }
      return std::begin(a)->value;
    }

    template <class Mapping, class N>
    typename Mapping::value_type::value_type value(Mapping a, N name) {
      auto pos = std::find_if(
          std::begin(a), std::end(a), [&name](const typename Mapping::value_type& t) { return (t.name == name); });
      if (pos != std::end(a)) {
        return pos->value;
      }
      return std::begin(a)->value;
    }

    template <class Mapping, class V>
    typename Mapping::value_type::name_type name_from_value(Mapping a, V value) {
      auto pos = std::find_if(
          std::begin(a), std::end(a), [&value](const typename Mapping::value_type& t) { return (t.value == value); });
      if (pos != std::end(a)) {
        return pos->name;
      }
      return std::begin(a)->name;
    }

  };  // namespace dd

  enum class DDSolidShape {
    dd_not_init = 0,
    ddbox = 1,
    ddtubs = 2,
    ddtrap = 3,
    ddcons = 4,
    ddpolycone = 5,
    ddpolyhedra = 6,
    ddunsupported1 = 7,
    ddunsupported2 = 8,
    ddtorus = 9,
    ddunion = 10,
    ddsubtraction = 11,
    ddintersection = 12,
    ddshapeless = 13,
    ddpseudotrap = 14,
    ddtrunctubs = 15,
    ddsphere = 16,
    ddellipticaltube = 17,
    ddcuttubs = 18,
    ddextrudedpolygon = 19,
    ddtrd1 = 20,
    ddtrd2 = 21,
    ddassembly = 22
  };

  const std::array<const cms::dd::NameValuePair<DDSolidShape>, 21> DDSolidShapeMap{
      {{DDSolidShape::dd_not_init, "Solid not initialized"},
       {DDSolidShape::ddbox, "Box"},
       {DDSolidShape::ddtubs, "Tube"},
       {DDSolidShape::ddtrap, "Trap"},
       {DDSolidShape::ddcons, "ConeSegment"},
       {DDSolidShape::ddpolycone, "Polycone"},
       {DDSolidShape::ddpolyhedra, "Polyhedra"},
       {DDSolidShape::ddtorus, "Torus"},
       {DDSolidShape::ddunion, "Union"},
       {DDSolidShape::ddsubtraction, "Subtraction"},
       {DDSolidShape::ddintersection, "Intersection"},
       {DDSolidShape::ddshapeless, "ShapelessSolid"},
       {DDSolidShape::ddpseudotrap, "PseudoTrap"},
       {DDSolidShape::ddtrunctubs, "TruncatedTube"},
       {DDSolidShape::ddsphere, "Sphere"},
       {DDSolidShape::ddellipticaltube, "EllipticalTube"},
       {DDSolidShape::ddcuttubs, "CutTube"},
       {DDSolidShape::ddextrudedpolygon, "ExtrudedPolygon"},
       {DDSolidShape::ddtrd1, "Trd1"},
       {DDSolidShape::ddtrd2, "Trd2"},
       {DDSolidShape::ddassembly, "Assembly"}}};

  const std::array<const cms::dd::ValuePair<LegacySolidShape, cms::DDSolidShape>, 21> LegacySolidShapeMap{
      {{LegacySolidShape::dd_not_init, cms::DDSolidShape::dd_not_init},
       {LegacySolidShape::ddbox, cms::DDSolidShape::ddbox},
       {LegacySolidShape::ddtubs, cms::DDSolidShape::ddtubs},
       {LegacySolidShape::ddtrap, cms::DDSolidShape::ddtrap},
       {LegacySolidShape::ddcons, cms::DDSolidShape::ddcons},
       {LegacySolidShape::ddpolycone_rz, cms::DDSolidShape::ddpolycone},
       {LegacySolidShape::ddpolycone_rrz, cms::DDSolidShape::ddpolycone},
       {LegacySolidShape::ddpolyhedra_rz, cms::DDSolidShape::ddpolyhedra},
       {LegacySolidShape::ddpolyhedra_rrz, cms::DDSolidShape::ddpolyhedra},
       {LegacySolidShape::ddtorus, cms::DDSolidShape::ddtorus},
       {LegacySolidShape::ddunion, cms::DDSolidShape::ddunion},
       {LegacySolidShape::ddsubtraction, cms::DDSolidShape::ddsubtraction},
       {LegacySolidShape::ddintersection, cms::DDSolidShape::ddintersection},
       {LegacySolidShape::ddshapeless, cms::DDSolidShape::ddshapeless},
       {LegacySolidShape::ddpseudotrap, cms::DDSolidShape::ddpseudotrap},
       {LegacySolidShape::ddtrunctubs, cms::DDSolidShape::ddtrunctubs},
       {LegacySolidShape::ddsphere, cms::DDSolidShape::ddsphere},
       {LegacySolidShape::ddellipticaltube, cms::DDSolidShape::ddellipticaltube},
       {LegacySolidShape::ddcuttubs, cms::DDSolidShape::ddcuttubs},
       {LegacySolidShape::ddextrudedpolygon, cms::DDSolidShape::ddextrudedpolygon},
       {LegacySolidShape::ddassembly, cms::DDSolidShape::ddassembly}}};

}  // namespace cms

#endif
