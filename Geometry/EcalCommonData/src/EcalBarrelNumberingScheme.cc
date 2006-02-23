///////////////////////////////////////////////////////////////////////////////
// File: EcalBarrelNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <iostream>

EcalBarrelNumberingScheme::EcalBarrelNumberingScheme(int iv) : 
  EcalNumberingScheme(iv) {
  if (verbosity>0) 
    std::cout << "Creating EcalBarrelNumberingScheme" << std::endl;
}

EcalBarrelNumberingScheme::~EcalBarrelNumberingScheme() {
  if (verbosity>0) 
    std::cout << "Deleting EcalBarrelNumberingScheme" << std::endl;
}

uint32_t EcalBarrelNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const 
{
  if (baseNumber.getLevels()<1)
    {
      if (verbosity>0) 
	std::cout << "ECalBarrelNumberingScheme::getUnitID: No level found in EcalBaseNumber"
		  << " Returning 0" << std::endl;
      return 0;
    }

  int PVid  = baseNumber.getCopyNumber(0);
  int MVid  = 1; 
  int MMVid = 1;

  if (baseNumber.getLevels() > 1) {
    MVid = baseNumber.getCopyNumber(1);
  } else { 
    if (verbosity>0) 
      std::cout << "ECalBarrelNumberingScheme::getUnitID: NullA pointer to "
		<< "alveole ! Use default id=1 " << std::endl;
  }
  if (baseNumber.getLevels() > 2) { 
    MMVid = baseNumber.getCopyNumber(2);
  } else { 
    if (verbosity>0) 
      std::cout << "ECalBarrelNumberingScheme::getUnitID: Null pointer to "
		<< "module ! Use default id=1 " << std::endl;
  }

  // z side 
  int zside   = baseNumber.getCopyNumber("EREG");
  zside=2*(1-zside)+1;

  // eta index of in Lyon geometry
  int ieta = PVid%5;
  if( ieta == 0) {ieta = 5;}
  int eta = 5 * (int) ((float)(PVid - 1)/10.) + ieta;

  // phi index in Lyon geometry
  int isubm = 1 + (int) ((float)(PVid - 1)/5.);
  int iphi  = (isubm%2) == 0 ? 2: 1;
  int phi=-1;

  if (zside == 1)
    phi = (20*(18-MMVid) + 2*(10-MVid) + iphi + 20) % 360  ;
  else if (zside == -1)
    phi = (541 - (20*(18-MMVid) + 2*(10-MVid) + iphi) ) % 360  ;

  if (phi == 0) 
    phi = 360;

  //pack it into an integer
  // to be consistent with EBDetId convention
  //  zside=2*(1-zside)+1;
  uint32_t intindex = EBDetId(zside*eta,phi).rawId();

  if (verbosity>1) 
    std::cout << "EcalBarrelNumberingScheme zside = "  << zside << " eta = " 
	      << eta << " phi = " << phi << " packed index = 0x" << std::hex 
	      << intindex << std::dec << std::endl;
  return intindex;

}

