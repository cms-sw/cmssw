///////////////////////////////////////////////////////////////////////////////
// File: EcalShashlikNumberingScheme.cc
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/HGCalCommonData/interface/EcalShashlikNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"

#include <iostream>
#include <iomanip>

EcalShashlikNumberingScheme::EcalShashlikNumberingScheme() : 
  EcalNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Creating EcalShashlikNumberingScheme";
}

EcalShashlikNumberingScheme::~EcalShashlikNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Deleting EcalShashlikNumberingScheme";
}
uint32_t EcalShashlikNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const {
  std::cerr << "=================================" << std::endl
	    << "EcalShashlikNumberingScheme::getUnitID is not implemented" << std::endl
	    << "=================================" << std::endl;
  return 0;
}

