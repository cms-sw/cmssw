///////////////////////////////////////////////////////////////////////////////
// File: EcalEndcapNumberingScheme.cc
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <iostream>

EcalEndcapNumberingScheme::EcalEndcapNumberingScheme(int iv) : 
  EcalNumberingScheme(iv) {
  if (verbosity>0) 
    std::cout << "Creating EcalEndcapNumberingScheme" << std::endl;
}

EcalEndcapNumberingScheme::~EcalEndcapNumberingScheme() {
  if (verbosity>0) 
    std::cout << "Deleting EcalEndcapNumberingScheme" << std::endl;
}
uint32_t EcalEndcapNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const 
{
  if (baseNumber.getLevels()<1)
    {
      if (verbosity>0) 
	std::cout << "EalEndcaplNumberingScheme::getUnitID: No level found in EcalBaseNumber"
		  << " Returning 0" << std::endl;
      return 0;
    }


  int PVid = baseNumber.getCopyNumber(0);
  int MVid = 1;
  if (baseNumber.getLevels() > 1) 
    MVid = baseNumber.getCopyNumber(1);
  else 
    if (verbosity>0) 
      std::cout << "ECalEndcapNumberingScheme::getUnitID: Null pointer to "
		<< "alveole ! Use default id=1 " << std::endl;

  int zside   = baseNumber.getCopyNumber("EREG");
  zside=2*(1-zside)+1;
  int module_number  = MVid;
  int crystal_number  = PVid;

  uint32_t intindex = EEDetId(module_number,crystal_number,zside,EEDetId::SCCRYSTALMODE).rawId();
  
  if (verbosity>1) 
    std::cout << "EcalEndcapNumberingScheme: zside = "  << zside 
	      << " super crystal = " << module_number << " crystal = " << crystal_number
	      << " packed index = 0x" << std::hex << intindex << std::dec 
	      << std::endl;
  return intindex;

}

