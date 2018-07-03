#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"

std::ostream&
operator<<(std::ostream& os, const DDSolidShape s)
{
  return os << "\tDDSolidShape index: " << static_cast<int>(s) << ", name: " << DDSolidShapesName::name(s);
}

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
  ddShapeTypeNames.emplace_back("ddshapeless");
  ddShapeTypeNames.emplace_back("ddpseudotrap");
  ddShapeTypeNames.emplace_back("ddtrunctubs");
  ddShapeTypeNames.emplace_back("ddsphere");
  ddShapeTypeNames.emplace_back("ddellipticaltube");
  ddShapeTypeNames.emplace_back("ddcuttubs");
  ddShapeTypeNames.emplace_back("ddextrudedpolygon");

  DDSolidShapesName ssn;
  DDSolidShape ish(DDSolidShape::dd_not_init);
  int index = static_cast<int>(ish);
  std::cout << std::left << std::setfill(' ');
  for(  ; ish <= DDSolidShape::ddextrudedpolygon; index = static_cast<int>(ish)+1, ish = DDSolidShape(index)) {
    switch( index ) {
    case 0:
      std::cout	<< index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(1) << " " <<  DDSolidShape::dd_not_init;
      break;
    case 1:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(9) << "\t" << DDSolidShape::ddbox;
      break;
    case 2:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(3) << "\t" << DDSolidShape::ddtubs;
      break;
    case 3:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddtrap;
      break;
    case 4:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddcons;
      break;
    case 5:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(1) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddpolycone_rz;
      break;
    case 6:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(1) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddpolyhedra_rz;
      break;
    case 7:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(1) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddpolycone_rrz;
      break;
    case 8:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(1) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddpolyhedra_rrz;
      break;
    case 9:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(9) << "\t" <<  DDSolidShape::ddtorus;
      break;
    case 10:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddunion;
      break;
    case 11:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(1) << "\t" << ssn.name(ish) << std::setw(1) << " " <<  DDSolidShape::ddsubtraction;
      break;
    case 12:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(1) << "\t" << ssn.name(ish) << std::setw(1) << " " <<  DDSolidShape::ddintersection;
      break;
    case 13:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddshapeless;
      break;
    case 14:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddpseudotrap;
      break;
    case 15:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(1) << " " <<  DDSolidShape::ddtrunctubs;
      break;
    case 16:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddsphere;
      break;
    case 17:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(1) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddellipticaltube;
      break;
    case 18:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(9) << "\t" << ssn.name(ish) << std::setw(9) << "\t" <<  DDSolidShape::ddcuttubs;
      break;
    case 19:
      std::cout << index << ":" << std::setw(4) << "\t" << ddShapeTypeNames[index] << std::setw(1) << "\t" << ssn.name(ish) << std::setw(3) << "\t" <<  DDSolidShape::ddextrudedpolygon;
      break;
    default:
      std::cout << "ERROR! No such shape!";
      break;
    }
    std::cout << std::endl;
  }
  return EXIT_SUCCESS;
}
