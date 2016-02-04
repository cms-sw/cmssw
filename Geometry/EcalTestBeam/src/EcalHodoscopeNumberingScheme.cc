///////////////////////////////////////////////////////////////////////////////
// File: EcalHodoscopeNumberingScheme.cc
// Description: Numbering scheme for TB H4 hodoscope detector
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/EcalTestBeam/interface/EcalHodoscopeNumberingScheme.h"
#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"

#include <iostream>

EcalHodoscopeNumberingScheme::EcalHodoscopeNumberingScheme() : 
  EcalNumberingScheme() {

  edm::LogInfo("EcalTBGeom") << "Creating EcalHodoscopeNumberingScheme";
}

EcalHodoscopeNumberingScheme::~EcalHodoscopeNumberingScheme() {
  edm::LogInfo("EcalTBGeom") << "Deleting EcalHodoscopeNumberingScheme";
}

uint32_t 
EcalHodoscopeNumberingScheme::getUnitID( const EcalBaseNumber& baseNumber ) const
{
  
  int level = baseNumber.getLevels();
  uint32_t intIndex = 0; 
  if (level > 0) 
  {
    // depth index - plans and fibers
    if(baseNumber.getLevelName(0) == "FIBR") 
    {
       uint32_t iFibr;

       if (baseNumber.getCopyNumber(0) > 32)
	  iFibr = 2 * (baseNumber.getCopyNumber(0)-33);
       else 
	  iFibr = 2*baseNumber.getCopyNumber(0) - 1;

       const uint32_t iPlane = baseNumber.getCopyNumber(1)-1;

       LogDebug("EcalTBGeom") << "Fibr/plane " << iFibr << " " << iPlane;
       
       intIndex = HodoscopeDetId( iPlane, iFibr ).rawId() ;

       LogDebug("EcalTBGeom") << "Index for fiber volume " 
			      << baseNumber.getLevelName(0) 
			      << " in plane " 
			      << baseNumber.getLevelName(1) 
			      << " = " << intIndex;
    }
  }
  return intIndex;
}

