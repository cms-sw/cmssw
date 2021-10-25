#ifndef DETECTOR_DESCRIPTION_CORE_DD_SOLID_SHAPES_H
#define DETECTOR_DESCRIPTION_CORE_DD_SOLID_SHAPES_H

#include <iosfwd>

enum class DDSolidShape {
  dd_not_init = 0,
  ddbox = 1,
  ddtubs = 2,
  ddtrap = 3,
  ddcons = 4,
  ddpolycone_rz = 5,
  ddpolyhedra_rz = 6,
  ddpolycone_rrz = 7,
  ddpolyhedra_rrz = 8,
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
  ddassembly = 20,
};

std::ostream& operator<<(std::ostream& os, const DDSolidShape s);

struct DDSolidShapesName {
  static const char* const name(DDSolidShape s) {
    static const char* const _names[] = {"Solid not initialized",
                                         "Box",
                                         "Tube(section)",
                                         "Trapezoid",
                                         "Cone(section)",
                                         "Polycone_rz",
                                         "Polyhedra_rz",
                                         "Polycone_rrz",
                                         "Polyhedra_rrz",
                                         "Torus",
                                         "UnionSolid",
                                         "SubtractionSolid",
                                         "IntersectionSolid",
                                         "ShapelessSolid",
                                         "PseudoTrapezoid",
                                         "TruncatedTube(section)",
                                         "Sphere(section)",
                                         "EllipticalTube",
                                         "CutTubs",
                                         "ExtrudedPolygon",
                                         "Assembly"};

    return _names[static_cast<int>(s)];
  }
};

#endif
