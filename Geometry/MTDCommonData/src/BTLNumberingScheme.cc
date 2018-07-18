#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include <iostream>
#include <cstring>

//#define EDM_ML_DEBUG

BTLNumberingScheme::BTLNumberingScheme() : 
  MTDNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "Creating BTLNumberingScheme";
#endif  
}

BTLNumberingScheme::~BTLNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "Deleting BTLNumberingScheme";
#endif  
}

uint32_t BTLNumberingScheme::getUnitID(const MTDBaseNumber& baseNumber) const {

  const uint32_t nLevels ( baseNumber.getLevels() ) ;

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "BTLNumberingScheme geometry levels = " << nLevels;
#endif  

  if( 9 > nLevels )
  {   
     edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                << "Not enough levels found in MTDBaseNumber ( "
                                <<  nLevels 
                                << ") Returning 0" ;
     return 0;
  }

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << baseNumber.getLevelName(0) << ", "
                          << baseNumber.getLevelName(1) << ", "
                          << baseNumber.getLevelName(2) << ", "
                          << baseNumber.getLevelName(3) << ", "
                          << baseNumber.getLevelName(4) << ", "
                          << baseNumber.getLevelName(5) << ", "
                          << baseNumber.getLevelName(6) << ", "
                          << baseNumber.getLevelName(7) << ", "
                          << baseNumber.getLevelName(8);
#endif

  const uint32_t crystal ( baseNumber.getCopyNumber( 0 ) ) ;

  const uint32_t modCopy ( baseNumber.getCopyNumber( 2 ) ) ;
  const uint32_t rodCopy ( baseNumber.getCopyNumber( 3 ) ) ;

  const std::string& modName ( baseNumber.getLevelName( 2 ) ) ; // name of module volume
  uint32_t pos = modName.find("Positive");

  const uint32_t zside = ( pos <= strlen(modName.c_str()) ? 1 : 0 );
  std::string baseName = modName.substr(modName.find(":")+1);
  const int modtyp ( ::atoi( (baseName.substr(8,1)).c_str() ) == 9 ?
                     ::atoi( (baseName.substr(7,1)).c_str() ) + 1 : ::atoi( (baseName.substr(7,1)).c_str() ) ) ;
  
  // error checking

  if( 1 > crystal ||
    64 < crystal ) 
  {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad crystal number = " 
                               << crystal 
                               << ", Volume Number = " 
                               << baseNumber.getCopyNumber(0);
    return 0 ;
  }
  
  if( 1  > modtyp ||
      3 < modtyp    )
  {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad module name = " 
                               << modName 
                               << ", Volume Name = " 
                               << baseNumber.getLevelName(2)              ;
    return 0 ;
  }
  
  if( 1 > modCopy ||
      54 < modCopy    )
    {
      edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                 << "****************** Bad module copy = " 
                                 << modCopy 
                                 << ", Volume Number = " 
                                 << baseNumber.getCopyNumber(2)              ;
      return 0 ;
    }
  
  if( 1 > rodCopy ||
      36 < rodCopy    )
    {
      edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                 << "****************** Bad rod copy = " 
                                 << rodCopy  
                                 << ", Volume Number = " 
                                 << baseNumber.getCopyNumber(4)              ;
      return 0 ;
    }
  
  // all inputs are fine. Go ahead and decode
  
  BTLDetId thisBTLdetid( zside, rodCopy, modCopy, modtyp, crystal );
  const int32_t intindex = thisBTLdetid.rawId() ;

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "BTL Numbering scheme: " 
                          << " rod = " << rodCopy
                          << " zside = " << zside
                          << " module = " << modCopy
                          << " modtyp = " << modtyp
                          << " crystal = " << crystal
                          << " Raw Id = " << intindex
                          << thisBTLdetid;
#endif

  return intindex ;  
}
