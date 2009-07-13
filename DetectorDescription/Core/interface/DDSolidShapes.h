#ifndef DDSolidShapes_h
#define DDSolidShapes_h
#include "DetectorDescription/Base/interface/DDException.h"

enum DDSolidShape { dd_not_init,
                    ddbox, ddtubs, ddtrap, ddcons,
                    ddpolycone_rz, ddpolyhedra_rz,
		    ddpolycone_rrz, ddpolyhedra_rrz,
		    ddtorus,
                    ddunion, ddsubtraction, ddintersection,
		    ddreflected,
		    ddshapeless,
		    ddpseudotrap, ddtrunctubs, ddsphere
		   };
		   
struct DDSolidShapesName {

  static const char * name(DDSolidShape s) 
  {
    static const char* c[] = { 
      "Solid not initialized",
      "Box", "Tube(section)", "Trapezoid", "Cone(section)",
      "Polycone_rz", "Polyhedra_rz",
      "Polycone_rrz", "Polyhedra_rrz",
      "Torus",
      "UnionSolid", "SubtractionSolid", "IntersectionSolid",
      "ReflectedSolid", 
      "ShapelessSolid",
      "PseudoTrapezoid","TruncatedTube(section)",
      "Sphere(section)"
    };
    return c[s];   			  
  }
  
  static const DDSolidShape index( const int& ind ) {
    switch (ind) {
    case 0:
      return dd_not_init;
      break;
    case 1:
      return ddbox;
      break;
    case 2:
      return ddtubs;
      break;
    case 3:
      return ddtrap;
      break;
    case 4:
      return ddcons;
      break;
    case 5:
      return ddpolycone_rz;
      break;
    case 6:
      return ddpolyhedra_rz;
      break;
    case 7:
      return ddpolycone_rrz;
      break;
    case 8:
      return ddpolyhedra_rrz;
      break;
    case 9:
      return ddtorus;
      break;
    case 10:
      return ddunion;
      break;
    case 11:
      return ddsubtraction;
      break;
    case 12:
      return ddintersection;
      break;
    case 13:
      return ddreflected;
      break;
    case 14:
      return ddshapeless;
      break;
    case 15:
      return ddpseudotrap;
      break;
    case 16:
      return ddtrunctubs;
      break;
    case 17:
      return ddsphere;
      break;
    default:
      throw DDException("DDSolidShapes:index wrong shape");   
      break;
    }
  }

};

		   

#endif
