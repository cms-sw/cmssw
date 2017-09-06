#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"

int main(int /*argc*/, char **/*argv[]*/)
{
  std::cout  << "" << std::endl;
  std::vector<std::string> ddShapeTypeNames;
  ddShapeTypeNames.emplace_back("dd_not_init");
  ddShapeTypeNames.emplace_back("ddbox"); 
  ddShapeTypeNames.emplace_back("ddtubs"); 
  ddShapeTypeNames.emplace_back("ddtrap"); 
  ddShapeTypeNames.emplace_back("ddcons");
  ddShapeTypeNames.emplace_back("ddpolycone_rz"); 
  ddShapeTypeNames.emplace_back("ddpolyhedra_rz");
  ddShapeTypeNames.emplace_back("ddpolycone_rrz"); 
  ddShapeTypeNames.emplace_back("ddpolyhedra_rrz");
  ddShapeTypeNames.emplace_back("ddtorus");
  ddShapeTypeNames.emplace_back("ddunion"); 
  ddShapeTypeNames.emplace_back("ddsubtraction"); 
  ddShapeTypeNames.emplace_back("ddintersection");
  ddShapeTypeNames.emplace_back("ddreflected");
  ddShapeTypeNames.emplace_back("ddshapeless");
  ddShapeTypeNames.emplace_back("ddpseudotrap");
  ddShapeTypeNames.emplace_back("ddtrunctubs");
  ddShapeTypeNames.emplace_back("ddsphere");
  ddShapeTypeNames.emplace_back("ddorb");
  ddShapeTypeNames.emplace_back("ddellipticaltube");
  ddShapeTypeNames.emplace_back("ddellipsoid");
  ddShapeTypeNames.emplace_back("ddparallelepiped");
  ddShapeTypeNames.emplace_back("ddcuttubs");
  ddShapeTypeNames.emplace_back("ddextrudedpolygon");
  ddShapeTypeNames.emplace_back("ddmultiunion");

  DDSolidShapesName ssn;
  DDSolidShape ish(dd_not_init);
  for ( ; ish <= ddextrudedpolygon; ish=DDSolidShape(ish+1) ) {
    switch (ish) {
    case 0:
      std::cout << ddShapeTypeNames[0] << " " << ssn.name(ish) << " " <<  dd_not_init;
      break;
    case 1:
      std::cout << ddShapeTypeNames[1] << " " << ssn.name(ish) << " " << ddbox;
      break;
    case 2:
      std::cout << ddShapeTypeNames[2] << " " << ssn.name(ish) << " " << ddtubs;
      break;
    case 3:
      std::cout << ddShapeTypeNames[3] << " " << ssn.name(ish) << " " <<  ddtrap;
      break;
    case 4:
      std::cout << ddShapeTypeNames[4] << " " << ssn.name(ish) << " " <<  ddcons;
      break;
    case 5:
      std::cout << ddShapeTypeNames[5] << " " << ssn.name(ish) << " " <<  ddpolycone_rz;
      break;
    case 6:
      std::cout << ddShapeTypeNames[6] << " " << ssn.name(ish) << " " <<  ddpolyhedra_rz;
      break;
    case 7:
      std::cout << ddShapeTypeNames[7] << " " << ssn.name(ish) << " " <<  ddpolycone_rrz;
      break;
    case 8:
      std::cout << ddShapeTypeNames[8] << " " << ssn.name(ish) << " " <<  ddpolyhedra_rrz;
      break;
    case 9:
      std::cout << ddShapeTypeNames[9] << " " << ssn.name(ish) << " " <<  ddtorus;
      break;
    case 10:
      std::cout << ddShapeTypeNames[10] << " " << ssn.name(ish) << " " <<  ddunion;
      break;
    case 11:
      std::cout << ddShapeTypeNames[11] << " " << ssn.name(ish) << " " <<  ddsubtraction;
      break;
    case 12:
      std::cout << ddShapeTypeNames[12] << " " << ssn.name(ish) << " " <<  ddintersection;
      break;
    case 13:
      std::cout << ddShapeTypeNames[13] << " " << ssn.name(ish) << " " <<  ddreflected;
      break;
    case 14:
      std::cout << ddShapeTypeNames[14] << " " << ssn.name(ish) << " " <<  ddshapeless;
      break;
    case 15:
      std::cout << ddShapeTypeNames[15] << " " << ssn.name(ish) << " " <<  ddpseudotrap;
      break;
    case 16:
      std::cout << ddShapeTypeNames[16] << " " << ssn.name(ish) << " " <<  ddtrunctubs;
      break;
    case 17:
      std::cout << ddShapeTypeNames[17] << " " << ssn.name(ish) << " " <<  ddsphere;
      break;
    case 18:
      std::cout << ddShapeTypeNames[18] << " " << ssn.name(ish) << " " <<  ddorb;
      break;
    case 19:
      std::cout << ddShapeTypeNames[19] << " " << ssn.name(ish) << " " <<  ddellipticaltube;
      break;
    case 20:
      std::cout << ddShapeTypeNames[20] << " " << ssn.name(ish) << " " <<  ddellipsoid;
      break;
    case 21:
      std::cout << ddShapeTypeNames[21] << " " << ssn.name(ish) << " " <<  ddparallelepiped;
      break;
    case 22:
      std::cout << ddShapeTypeNames[22] << " " << ssn.name(ish) << " " <<  ddcuttubs;
      break;
    case 23:
      std::cout << ddShapeTypeNames[23] << " " << ssn.name(ish) << " " <<  ddextrudedpolygon;
      break;
    case 24:
      std::cout << ddShapeTypeNames[24] << " " << ssn.name(ish) << " " <<  ddmultiunion;
      break;
    default:
      std::cout << "ERROR! No such shape!";
      break;
    }
    std::cout << std::endl;
  }
  return EXIT_SUCCESS;
}
