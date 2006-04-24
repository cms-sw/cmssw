///////////////////////////////////////////////////////////////////////////////
// File: EcalEndcapNumberingScheme.cc
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <iostream>

EcalEndcapNumberingScheme::EcalEndcapNumberingScheme() : 
  EcalNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Creating EcalEndcapNumberingScheme";
}

EcalEndcapNumberingScheme::~EcalEndcapNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Deleting EcalEndcapNumberingScheme";
}
uint32_t EcalEndcapNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const {
  if (baseNumber.getLevels()<1) {
    edm::LogWarning("EcalGeom") << "EalEndcaplNumberingScheme::getUnitID: No "
				<< "level found in EcalBaseNumber Returning 0";
    return 0;
  }

  int PVid = baseNumber.getCopyNumber(0);
  int MVid = 1;
  if (baseNumber.getLevels() > 1) 
    MVid = baseNumber.getCopyNumber(1);
  else 
    edm::LogWarning("EcalGeom") << "ECalEndcapNumberingScheme::getUnitID: Null"
				<< " pointer to alveole ! Use default id=1";

  int zside   = baseNumber.getCopyNumber("EREG");
  zside=2*(1-zside)+1;
  int module_number  = MVid;
  int crystal_number  = PVid;

  uint32_t intindex = EEDetId(module_number,crystal_number,zside,EEDetId::SCCRYSTALMODE).rawId();
  
  LogDebug("EcalGeom") << "EcalEndcapNumberingScheme: zside = "  << zside 
		       << " super crystal = " << module_number << " crystal = "
		       << crystal_number << " packed index = 0x" << std::hex 
		       << intindex << std::dec;
  return intindex;

}

