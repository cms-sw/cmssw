/***************************************************************************
                          testShapes.cpp  -  description
                             -------------------
    Author               : Michael Case
    email                : case@physics.ucdavis.edu

    Last Updated         : May 29, 2007
 ***************************************************************************/
#include <iostream>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

int main(int /*argc*/, char **/*argv[]*/)
{
  std::cout  << "" << std::endl;
  std::vector<std::string> ddShapeTypeNames;
  ddShapeTypeNames.push_back("dd_not_init");
  ddShapeTypeNames.push_back("ddbox"); 
  ddShapeTypeNames.push_back("ddtubs"); 
  ddShapeTypeNames.push_back("ddtrap"); 
  ddShapeTypeNames.push_back("ddcons");
  ddShapeTypeNames.push_back("ddpolycone_rz"); 
  ddShapeTypeNames.push_back("ddpolyhedra_rz");
  ddShapeTypeNames.push_back("ddpolycone_rrz"); 
  ddShapeTypeNames.push_back("ddpolyhedra_rrz");
  ddShapeTypeNames.push_back("ddtorus");
  ddShapeTypeNames.push_back("ddunion"); 
  ddShapeTypeNames.push_back("ddsubtraction"); 
  ddShapeTypeNames.push_back("ddintersection");
  ddShapeTypeNames.push_back("ddreflected");
  ddShapeTypeNames.push_back("ddshapeless");
  ddShapeTypeNames.push_back("ddpseudotrap");
  ddShapeTypeNames.push_back("ddtrunctubs");
  ddShapeTypeNames.push_back("ddsphere");
  ddShapeTypeNames.push_back("ddorb");
  ddShapeTypeNames.push_back("ddellipticaltube");
  ddShapeTypeNames.push_back("ddellipsoid");

  DDSolidShapesName ssn;
  DDSolidShape ish(dd_not_init);
  for ( ; ish <= ddellipsoid; ish=DDSolidShape(ish+1) ) {
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
    default:
      std::cout << "ERROR! No such shape!";
      break;
    }
    std::cout << std::endl;
  }
  return EXIT_SUCCESS;
}
