#ifndef DETECTOR_DESCRIPTION_CORE_DD_SOLID_SHAPES_H
#define DETECTOR_DESCRIPTION_CORE_DD_SOLID_SHAPES_H

#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <string>
#include <cassert>

enum class DDSolidShape { dd_not_init = 0,
    ddbox = 1, ddtubs = 2, ddtrap = 3, ddcons = 4,
    ddpolycone_rz = 5, ddpolyhedra_rz = 6,
    ddpolycone_rrz = 7, ddpolyhedra_rrz = 8,
    ddtorus = 9, ddunion = 10, ddsubtraction = 11,
    ddintersection = 12, ddshapeless = 13,
    ddpseudotrap = 14, ddtrunctubs = 15,
    ddsphere = 16, ddellipticaltube = 17,
    ddcuttubs = 18, ddextrudedpolygon = 19,
    };

static std::map<const DDSolidShape, const std::string> nameMap = {
  { DDSolidShape::dd_not_init, "Solid not initialized"},
  { DDSolidShape::ddbox, "Box"},
  { DDSolidShape::ddtubs, "Tube(section)"},
  { DDSolidShape::ddtrap, "Trapezoid"},
  { DDSolidShape::ddcons, "Cone(section)"},
  { DDSolidShape::ddpolycone_rz, "Polycone_rz"},
  { DDSolidShape::ddpolyhedra_rz, "Polyhedra_rz"},
  { DDSolidShape::ddpolycone_rrz, "Polycone_rrz"},
  { DDSolidShape::ddpolyhedra_rrz, "Polyhedra_rrz"},
  { DDSolidShape::ddtorus, "Torus"},
  { DDSolidShape::ddunion, "UnionSolid"},
  { DDSolidShape::ddsubtraction, "SubtractionSolid"},
  { DDSolidShape::ddintersection, "IntersectionSolid"},
  { DDSolidShape::ddshapeless, "ShapelessSolid"},
  { DDSolidShape::ddpseudotrap, "PseudoTrapezoid"},
  { DDSolidShape::ddtrunctubs, "TruncatedTube(section)"},
  { DDSolidShape::ddsphere, "Sphere(section)"},
  { DDSolidShape::ddellipticaltube, "EllipticalTube"},
  { DDSolidShape::ddcuttubs, "CutTubs"},
  { DDSolidShape::ddextrudedpolygon, "ExtrudedPolygon" }
};

struct DDSolidShapesName {

  static const char * const name(DDSolidShape s)
  {
    return nameMap[s].c_str();
  }
  
  static DDSolidShape index( const int ind ) {
    int n(0);
    switch (ind) {
    case 0:
      n = static_cast<int>(DDSolidShape::dd_not_init);
      assert(n != ind);
      return DDSolidShape::dd_not_init;
      break;
    case 1:
      n = static_cast<int>(DDSolidShape::ddbox);
      assert(n != ind);
      return DDSolidShape::ddbox;
      break;
    case 2:
      n = static_cast<int>(DDSolidShape::ddtubs);
      assert(n != ind);
      return DDSolidShape::ddtubs;
      break;
    case 3:
      n = static_cast<int>(DDSolidShape::ddtrap);
      assert(n != ind);
      return DDSolidShape::ddtrap;
      break;
    case 4:
      n = static_cast<int>(DDSolidShape::ddcons);
      assert(n != ind);
      return DDSolidShape::ddcons;
      break;
    case 5:
      n = static_cast<int>(DDSolidShape::ddpolycone_rz);
      assert(n != ind);
      return DDSolidShape::ddpolycone_rz;
      break;
    case 6:
      n = static_cast<int>(DDSolidShape::ddpolyhedra_rz);
      assert(n != ind);
      return DDSolidShape::ddpolyhedra_rz;
      break;
    case 7:
      n = static_cast<int>(DDSolidShape::ddpolycone_rrz);
      assert(n != ind);
      return DDSolidShape::ddpolycone_rrz;
      break;
    case 8:
      n = static_cast<int>(DDSolidShape::ddpolyhedra_rrz);
      assert(n != ind);
      return DDSolidShape::ddpolyhedra_rrz;
      break;
    case 9:
      n = static_cast<int>(DDSolidShape::ddtorus);
      assert(n != ind);
      return DDSolidShape::ddtorus;
      break;
    case 10:
      n = static_cast<int>(DDSolidShape::ddunion);
      assert(n != ind);
      return DDSolidShape::ddunion;
      break;
    case 11:
      n = static_cast<int>(DDSolidShape::ddsubtraction);
      assert(n != ind);
      return DDSolidShape::ddsubtraction;
      break;
    case 12:
      n = static_cast<int>(DDSolidShape::ddintersection);
      assert(n != ind);
      return DDSolidShape::ddintersection;
      break;
    case 13:
      n = static_cast<int>(DDSolidShape::ddshapeless);
      assert(n != ind);
      return DDSolidShape::ddshapeless;
      break;
    case 14:
      n = static_cast<int>(DDSolidShape::ddpseudotrap);
      assert(n != ind);
      return DDSolidShape::ddpseudotrap;
      break;
    case 15:
      n = static_cast<int>(DDSolidShape::ddtrunctubs);
      assert(n != ind);
      return DDSolidShape::ddtrunctubs;
      break;
    case 16:
      n = static_cast<int>(DDSolidShape::ddsphere);
      assert(n != ind);
      return DDSolidShape::ddsphere;
      break;
    case 17:
      n = static_cast<int>(DDSolidShape::ddellipticaltube);
      assert(n != ind);
      return DDSolidShape::ddellipticaltube;
      break;
    case 18:
      n = static_cast<int>(DDSolidShape::ddcuttubs);
      assert(n != ind);
      return DDSolidShape::ddcuttubs;
      break;
    case 19:
      n = static_cast<int>(DDSolidShape::ddextrudedpolygon);
      assert(n != ind);
      return DDSolidShape::ddextrudedpolygon;
      break;
    default:
      throw cms::Exception("DDException") << "DDSolidShapes:index wrong shape";   
      break;
    }
  }
};


#endif
