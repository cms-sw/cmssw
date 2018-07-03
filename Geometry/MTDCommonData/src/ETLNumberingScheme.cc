#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include <iostream>

//#define EDM_ML_DEBUG

ETLNumberingScheme::ETLNumberingScheme() : 
  MTDNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "Creating ETLNumberingScheme";
#endif  
}

ETLNumberingScheme::~ETLNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "Deleting ETLNumberingScheme";
#endif  
}

uint32_t ETLNumberingScheme::getUnitID(const MTDBaseNumber& baseNumber) const {

  const uint32_t nLevels ( baseNumber.getLevels() ) ;

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "ETLNumberingScheme geometry levels = " << nLevels;
#endif
  

  if( 11 > nLevels )
  {   
     edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                                << "Not enough levels found in MTDBaseNumber ( "
                                <<  nLevels 
                                << ") Returning 0" ;
     return 0;
  }

  const uint32_t modCopy ( baseNumber.getCopyNumber( 2 ) ) ;

  const std::string& ringName ( baseNumber.getLevelName( 3 ) ) ; // name of ring volume
  const int modtyp(0) ;
  std::string baseName = ringName.substr(ringName.find(":")+1);
  const int ringCopy ( ::atoi( baseName.c_str() + 4 ) );

  const uint32_t sideCopy ( baseNumber.getCopyNumber( 7 ) ) ;
  const uint32_t zside ( sideCopy == 1 ? 1 : 0 ) ;

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << baseNumber.getLevelName(0) << ", "
                          << baseNumber.getLevelName(1) << ", "
                          << baseNumber.getLevelName(2) << ", "
                          << baseNumber.getLevelName(3) << ", "
                          << baseNumber.getLevelName(4) << ", "
                          << baseNumber.getLevelName(5) << ", "
                          << baseNumber.getLevelName(6) << ", "
                          << baseNumber.getLevelName(7) << ", "
                          << baseNumber.getLevelName(8) << ", "
                          << baseNumber.getLevelName(9) << ", "
                          << baseNumber.getLevelName(10) << ", "
                          << baseNumber.getLevelName(11)         ;
#endif
  
  // error checking

  if( modtyp != 0 ) 
  {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad module name = " 
                               << modtyp 
                               << ", Volume Name = " 
                               << baseNumber.getLevelName(4)              ;
    return 0 ;
  }
  
  if( 1 > modCopy ||
      176 < modCopy    )
    {
      edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                                 << "****************** Bad module copy = " 
                                 << modCopy 
                                 << ", Volume Number = " 
                                 << baseNumber.getCopyNumber(4)              ;
      return 0 ;
    }
  
  if( 1 > ringCopy ||
      11 < ringCopy    )
    {
      edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                                 << "****************** Bad ring copy = " 
                                 << ringCopy  
                                 << ", Volume Number = " 
                                 << baseNumber.getCopyNumber(3)              ;
      return 0 ;
    }
  
  // all inputs are fine. Go ahead and decode
  
  ETLDetId thisETLdetid( zside, ringCopy, modCopy, modtyp );
  const int32_t intindex = thisETLdetid.rawId() ;

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "ETL Numbering scheme: " 
                          << " ring = " << ringCopy
                          << " zside = " << zside
                          << " module = " << modCopy
                          << " modtyp = " << modtyp
                          << " Raw Id = " << intindex
                          << thisETLdetid;
#endif

  return intindex ;  
}
