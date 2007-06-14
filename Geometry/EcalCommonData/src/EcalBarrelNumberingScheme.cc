///////////////////////////////////////////////////////////////////////////////
// File: EcalBarrelNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <iostream>

EcalBarrelNumberingScheme::EcalBarrelNumberingScheme() : 
  EcalNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Creating EcalBarrelNumberingScheme";
}

EcalBarrelNumberingScheme::~EcalBarrelNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Deleting EcalBarrelNumberingScheme";
}

uint32_t EcalBarrelNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const {

  const uint32_t nLevels ( baseNumber.getLevels() ) ;

  LogDebug("EcalGeom") << "ECalBarrelNumberingScheme geometry levels = " << nLevels;
  
  if( 7 > nLevels )
    {
      
      edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                  << "Not enough levels found in EcalBaseNumber ( "
                                  <<  nLevels 
                                  << ") Returning 0" ;
      return 0;
    }
  
  // Static geometry

  if ( nLevels <= 8 ) {
    
    
    if (baseNumber.getLevels()<1) {
      edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID: No "
                                  << "level found in EcalBaseNumber Returning 0";
      return 0;
    }
    
    int PVid  = baseNumber.getCopyNumber(0);
    int MVid  = 1; 
    int MMVid = 1;
    
    if (baseNumber.getLevels() > 1) {
      MVid = baseNumber.getCopyNumber(1);
    } else { 
      edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID: Null"
                                  << " pointer to alveole ! Use default id=1";  }
    if (baseNumber.getLevels() > 2) { 
      MMVid = baseNumber.getCopyNumber(2);
    } else { 
      edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID: Null"
                                  << " pointer to module ! Use default id=1";
    }
    
    // z side 
    int zside   = baseNumber.getCopyNumber("EREG");
    if ( zside == 1 || zside == 2 ) {
      zside=2*(1-zside)+1;
    }
    else if ( zside == 0 ) {
      // MTCC geometry
      int zMTCC = baseNumber.getCopyNumber("EREG_P");
      if ( zMTCC == 1 ) {
        zside = 1;
      }
    }
    
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

    LogDebug("EcalGeom") << "EcalBarrelNumberingScheme zside = "  << zside 
                         << " eta = " << eta << " phi = " << phi 
                         << " packed index = 0x" << std::hex << intindex 
                         << std::dec;
    return intindex;
  
  }
    
  // Algorithmic geometry
  
  else
    
    {
      
      const std::string & cryName = baseNumber.getLevelName( 0 ) ; // name of crystal volume
      //std::istringstream stream ( cryName.substr( cryName.find_first_of('_') + 1, 2 ) ) ;
      //int cryType ( -1 ) ;
      // stream >> cryType ;
      int cryType = ::atoi( cryName.substr( cryName.find_first_of('_') + 1, 2 ).c_str());

      const uint32_t wallCopy ( baseNumber.getCopyNumber( 3 ) ) ;
      const uint32_t hawCopy  ( baseNumber.getCopyNumber( 4 ) ) ;
      const uint32_t fawCopy  ( baseNumber.getCopyNumber( 5 ) ) ;
      const uint32_t supmCopy ( baseNumber.getCopyNumber( 6 ) ) ;


      LogDebug("EcalGeom") << baseNumber.getLevelName(0) << ", "
                           << baseNumber.getLevelName(1) << ", "
                           << baseNumber.getLevelName(2) << ", "
                           << baseNumber.getLevelName(3) << ", "
                           << baseNumber.getLevelName(4) << ", "
                           << baseNumber.getLevelName(5) << ", "
                           << baseNumber.getLevelName(6) << ", "
                           << baseNumber.getLevelName(7)         ;

      // error checking

      if( 1  > cryType ||
          17 < cryType    )
        {
          edm::LogWarning("EdalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                      << "****************** Bad crystal name = " << cryName 
                                      << ", Volume Name = " << baseNumber.getLevelName(0)              ;
          return 0 ;
        }

      if( 1 > wallCopy ||
          5 < wallCopy    )
        {
          edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                      << "****************** Bad wall copy = " << wallCopy 
                                      << ", Volume Name = " << baseNumber.getLevelName(3)              ;
          return 0 ;
        }

      if( 1 > hawCopy ||
          2 < hawCopy    )
        {
          edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                      << "****************** Bad haw copy = " << hawCopy  
                                      << ", Volume Name = " << baseNumber.getLevelName(4)              ;
          return 0 ;
        }

      if( 1  > fawCopy ||
          10 < fawCopy    )
        {
          edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                      << "****************** Bad faw copy = " << fawCopy  
                                      << ", Volume Name = " << baseNumber.getLevelName(5)              ;
          return 0 ;
        }

      if( 1  > supmCopy ||
          36 < supmCopy    )
        {
          edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                      << "****************** Bad supermodule copy = " << supmCopy 
                                      << ", Volume Name = " << baseNumber.getLevelName(6)              ;
          return 0 ;
        }

      // all inputs are fine. Go ahead and decode

      const int32_t zsign    ( 18 < supmCopy ? -1 : 1 ) ;

      const int32_t eta      ( 5*( cryType - 1 ) + wallCopy ) ;

      const int32_t phi      ( 18 < supmCopy ?
                               20*( supmCopy - 19 ) + 2*( 10 - fawCopy ) + 3 - hawCopy :
                               20*( supmCopy -  1 ) + 2*( fawCopy - 1  ) +     hawCopy   ) ;

      const int32_t intindex ( EBDetId( zsign*eta, phi ).rawId() ) ;


      LogDebug("EcalGeom") << "EcalBarrelNumberingScheme: "
                           << "supmCopy = " << supmCopy
                           << ", fawCopy = " << fawCopy
                           << ", hawCopy = " << hawCopy
                           << ", wallCopy = " << wallCopy
                           << ", cryType = " << cryType
                           << "\n           zsign = "  << zsign 
                           << ", eta = " << eta 
                           << ", phi = " << phi 
                           << ", packed index = 0x" << std::hex << intindex << std::dec ;

      return intindex;
    }
  
}
