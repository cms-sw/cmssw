#ifndef DETECTOR_DESCRIPTION_DDCMS_DD_SOLID_SHAPES_H
#define DETECTOR_DESCRIPTION_DDCMS_DD_SOLID_SHAPES_H

// Keep this include until DD4hep migration is complete.
// Some migrated code will not compile without the old DDSolidShape.
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

#include <iosfwd>
#include <array>
#include <algorithm>

namespace cms {

  namespace dd {
    template <class T>
    struct NameValuePair {
      using value_type = T;
      const T value;
      const char* const name;
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
  };  // namespace dd

  // This version of DDSolidShape must be commented out until DD4hep migration is complete.
  // Having two versions of DDSolidShape will cause compile errors for some migrated code.
  /*
  enum class DDSolidShape {
    dd_not_init = 0,
    ddbox = 1,
    ddtubs = 2,
    ddtrap = 3,
    ddcons = 4,
    ddpolycone = 5,
    ddpolyhedra = 6,
    ddtorus = 7,
    ddunion = 8,
    ddsubtraction = 9,
    ddintersection = 10,
    ddshapeless = 11,
    ddpseudotrap = 12,
    ddtrunctubs = 13,
    ddsphere = 14,
    ddellipticaltube = 15,
    ddcuttubs = 16,
    ddextrudedpolygon = 17,
  };
  */

  const std::array<const cms::dd::NameValuePair<DDSolidShape>, 18> DDSolidShapeMap{
      {{DDSolidShape::dd_not_init, "Solid not initialized"},
       {DDSolidShape::ddbox, "Box"},
       {DDSolidShape::ddtubs, "Tube"},
       {DDSolidShape::ddtrap, "Trapezoid"},
       {DDSolidShape::ddcons, "Cone"},
       {DDSolidShape::ddpolycone_rz, "Polycone"},
       {DDSolidShape::ddpolyhedra_rz, "Polyhedra"},
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
       {DDSolidShape::ddextrudedpolygon, "ExtrudedPolygon"}}};
}  // namespace cms

#endif
