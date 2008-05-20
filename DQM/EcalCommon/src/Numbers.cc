// $Id: Numbers.cc,v 1.55 2008/04/22 10:21:55 dellaric Exp $

/*!
  \file Numbers.cc
  \brief Some "id" conversions
  \author B. Gobbo
  \version $Revision: 1.55 $
  \date $Date: 2008/04/22 10:21:55 $
*/

#include <sstream>
#include <iomanip>

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDetId/interface/EcalElectronicsId.h>
#include <DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h>
#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>

#include "FWCore/Framework/interface/NoRecordException.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"

//-------------------------------------------------------------------------

const EcalElectronicsMapping* Numbers::map = 0;

bool Numbers::init = false;

#define USEMAP false

//-------------------------------------------------------------------------

void Numbers::initGeometry( const edm::EventSetup& setup, bool verbose ) {

  if( Numbers::init ) return;

  if ( verbose ) std::cout << "Initializing EcalElectronicsMapping ..." << std::endl;

  Numbers::init = true;

  try {

    edm::ESHandle< EcalElectronicsMapping > handle;
    setup.get< EcalMappingRcd >().get(handle);
    Numbers::map = handle.product();

    if ( verbose ) std::cout << "done." << std::endl;

  } catch ( edm::eventsetup::NoRecordException< EcalMappingRcd > &e ) {

    if ( verbose ) std::cout << "NOT done." << std::endl;

  }

  if ( verbose ) std::cout << std::endl;

}

//-------------------------------------------------------------------------

int Numbers::iEB( const int ism ) throw( std::runtime_error ) {

  // EB-
  if( ism >=  1 && ism <= 18 ) return( -ism );

  // EB+
  if( ism >= 19 && ism <= 36 ) return( +ism - 18 );

  std::ostringstream s;
  s << "Wrong SM id determination: iSM = " << ism;
  throw( std::runtime_error( s.str() ) );

}

//-------------------------------------------------------------------------

std::string Numbers::sEB( const int ism  ) {

  int ieb = Numbers::iEB( ism );

  std::ostringstream s;
  s << "EB" << std::setw(3) << std::setfill('0')
    << std::setiosflags( std::ios::showpos )
    << std::setiosflags( std::ios::internal )
    << ieb
    << std::resetiosflags( std::ios::showpos )
    << std::resetiosflags( std::ios::internal );
  return( s.str() );

}

//-------------------------------------------------------------------------

int Numbers::iEE( const int ism ) throw( std::runtime_error ) {

  // EE-
  if( ism ==  1 ) return( -7 );
  if( ism ==  2 ) return( -8 );
  if( ism ==  3 ) return( -9 );
  if( ism ==  4 ) return( -1 );
  if( ism ==  5 ) return( -2 );
  if( ism ==  6 ) return( -3 );
  if( ism ==  7 ) return( -4 );
  if( ism ==  8 ) return( -5 );
  if( ism ==  9 ) return( -6 );

  // EE+
  if( ism == 10 ) return( +7 );
  if( ism == 11 ) return( +8 );
  if( ism == 12 ) return( +9 );
  if( ism == 13 ) return( +1 );
  if( ism == 14 ) return( +2 );
  if( ism == 15 ) return( +3 );
  if( ism == 16 ) return( +4 );
  if( ism == 17 ) return( +5 );
  if( ism == 18 ) return( +6 );

  std::ostringstream s;
  s << "Wrong SM id determination: iSM = " << ism;
  throw( std::runtime_error( s.str() ) );

}

//-------------------------------------------------------------------------

EcalSubdetector Numbers::subDet( const EBDetId& id ) {

  return( id.subdet() );

}

//-------------------------------------------------------------------------

EcalSubdetector Numbers::subDet( const EEDetId& id ) {

  return( id.subdet() );

}

//-------------------------------------------------------------------------

EcalSubdetector Numbers::subDet( const EcalTrigTowerDetId& id ) {

  return( id.subDet() );

}

//-------------------------------------------------------------------------

EcalSubdetector Numbers::subDet( const EcalElectronicsId& id ) {

  return( id.subdet() );

}

//-------------------------------------------------------------------------

EcalSubdetector Numbers::subDet( const EcalPnDiodeDetId& id ) {

  return( (EcalSubdetector) id.iEcalSubDetectorId() );

}

//-------------------------------------------------------------------------

EcalSubdetector Numbers::subDet( const EcalDCCHeaderBlock& id ) throw( std::runtime_error ) {

  int idcc = id.id();

  // EE-
  if ( idcc >=  1 && idcc <=  9 ) return( EcalEndcap );

  // EB-/EB+
  if ( idcc >= 10 && idcc <= 45 ) return( EcalBarrel);

  // EE+
  if ( idcc >= 46 && idcc <= 54 ) return( EcalEndcap );

  std::ostringstream s;
  s << "Wrong DCC id: dcc = " << idcc;
  throw( std::runtime_error( s.str() ) );

}

//-------------------------------------------------------------------------

std::string Numbers::sEE( const int ism  ) {

  int iee = Numbers::iEE( ism );

  std::ostringstream s;
  s << "EE" << std::setw(3) << std::setfill('0')
    << std::setiosflags( std::ios::showpos )
    << std::setiosflags( std::ios::internal )
    << iee
    << std::resetiosflags( std::ios::showpos )
    << std::resetiosflags( std::ios::internal );
  return( s.str() );

}

//-------------------------------------------------------------------------

int Numbers::iSM( const int ism, const EcalSubdetector subdet ) throw( std::runtime_error ) {

  if( subdet == EcalBarrel ) {

    // EB-
    if( ism >=  1 && ism <= 18 ) return( ism+18 );

    // EB+
    if( ism >= 19 && ism <= 36 ) return( ism-18 );

    std::ostringstream s;
    s << "Wrong SM id: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );

  } else if( subdet == EcalEndcap ) {

    // EE-
    if( ism >=  1 && ism <=  9 ) return( ism+9 );

    // EE+
    if (ism >= 10 && ism <= 18 ) return( ism-9 );

    std::ostringstream s;
    s << "Wrong SM id: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );

  } else {

    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( std::runtime_error( s.str() ) );

  }

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EBDetId& id ) throw( std::runtime_error ) {

  if( Numbers::map && USEMAP ) {

    EcalElectronicsId eid = Numbers::map->getElectronicsId(id);
    int idcc = eid.dccId();

    // EB-/EB+
    if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

    std::ostringstream s;
    s << "Wrong DCC id: dcc = " << idcc;
    throw( std::runtime_error( s.str() ) );

  } else {

    return( Numbers::iSM( id.ism(), EcalBarrel ) );

  }

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EEDetId& id ) throw( std::runtime_error ) {

  if( Numbers::map ) {

    EcalElectronicsId eid = Numbers::map->getElectronicsId(id);
    int idcc = eid.dccId();

    // EE-
    if( idcc >=  1 && idcc <=  9 ) return( idcc );

    // EE+
    if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

    std::ostringstream s;
    s << "Wrong DCC id: dcc = " << idcc;
    throw( std::runtime_error( s.str() ) );

  } else {

    std::ostringstream s;
    s << "ECAL Geometry not available";
    throw( std::runtime_error( s.str() ) );

  }

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalTrigTowerDetId& id ) throw( std::runtime_error ) {

  EcalSubdetector subdet = Numbers::subDet( id );

  if( subdet == EcalBarrel ) {

    if( Numbers::map && USEMAP ) {

      int idcc = Numbers::map->DCCid(id);

      // EB-/EB+
      if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

      std::ostringstream s;
      s << "Wrong DCC id: dcc = " << idcc;
      throw( std::runtime_error( s.str() ) );

    } else {

      return( Numbers::iSM( id.iDCC(), EcalBarrel ) );

    }

  } else if( subdet ==  EcalEndcap) {

    if( Numbers::map ) {

      int idcc = Numbers::map->DCCid(id);

      // EE-
      if( idcc >=  1 && idcc <=  9 ) return( idcc );

      // EE+
      if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

      std::ostringstream s;
      s << "Wrong DCC id: dcc = " << idcc;
      throw( std::runtime_error( s.str() ) );

    } else {

      std::ostringstream s;
      s << "ECAL Geometry not available";
      throw( std::runtime_error( s.str() ) );

    }

  } else {

    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( std::runtime_error( s.str() ) );

  }

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalElectronicsId& id ) throw( std::runtime_error ) {

  int idcc = id.dccId();

  // EE-
  if( idcc >=  1 && idcc <=  9 ) return( idcc );

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  // EE+
  if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

  std::ostringstream s;
  s << "Wrong DCC id: dcc = " << idcc;
  throw( std::runtime_error( s.str() ) );

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalPnDiodeDetId& id ) throw( std::runtime_error ) {

  int idcc = id.iDCCId();

  // EE-
  if( idcc >=  1 && idcc <=  9 ) return( idcc );

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  // EE+
  if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

  std::ostringstream s;
  s << "Wrong DCC id: dcc = " << idcc;
  throw( std::runtime_error( s.str() ) );

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalDCCHeaderBlock& id, const EcalSubdetector subdet ) throw( std::runtime_error ) {

  int idcc = id.id();

  // EE-
  if( idcc >=  1 && idcc <=  9 ) return( idcc );

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  // EE+
  if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

  std::ostringstream s;
  s << "Wrong DCC id: dcc = " << idcc;
  throw( std::runtime_error( s.str() ) );

}

//-------------------------------------------------------------------------

int Numbers::iTT( const int ism, const EcalSubdetector subdet, const int i1, const int i2 ) throw( std::runtime_error ) {

  if( subdet == EcalBarrel ) {

    int iet = 1 + ((i1-1)/5);
    int ipt = 1 + ((i2-1)/5);

    return( (ipt-1) + 4*(iet-1) + 1 );

  } else if( subdet == EcalEndcap ) {

    int iz = 0;

    if( ism >=  1 && ism <=  9 ) iz = -1;
    if( ism >= 10 && ism <= 18 ) iz = +1;

    if( EEDetId::validDetId(i1, i2, iz) ) {

      EEDetId id(i1, i2, iz, EEDetId::XYMODE);

      if( Numbers::iSM( id ) != ism ) return( -1 );

      if( Numbers::map ) {

        EcalElectronicsId eid = Numbers::map->getElectronicsId(id);

        return( eid.towerId() );

      } else {

        std::ostringstream s;
        s << "ECAL Geometry not available";
        throw( std::runtime_error( s.str() ) );

      }

    } else {

      return( -1 );

    }

  } else {

    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( std::runtime_error( s.str() ) );

  }

}

//-------------------------------------------------------------------------

int Numbers::iTT( const EcalTrigTowerDetId& id ) throw( std::runtime_error ) {

  EcalSubdetector subdet = Numbers::subDet( id );

  if( subdet == EcalBarrel ) {

    if( Numbers::map && USEMAP ) {

      return( Numbers::map->iTT(id) );

    } else {

      return( id.iTT() );

    }

  } else if( subdet ==  EcalEndcap) {

    if( Numbers::map ) {

      return( Numbers::map->iTT(id) );

    } else {

      std::ostringstream s;
      s << "ECAL Geometry not available";
      throw( std::runtime_error( s.str() ) );

    }

  } else {

    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( std::runtime_error( s.str() ) );

  }

}

//-------------------------------------------------------------------------

std::vector<DetId> Numbers::crystals( const EcalTrigTowerDetId& id ) throw( std::runtime_error ) {

  if( Numbers::map ) {

    int itcc = Numbers::map->TCCid(id);
    int itt = Numbers::map->iTT(id);

    return( Numbers::map->ttConstituents( itcc, itt ) );

  } else {

    std::ostringstream s;
    s << "ECAL Geometry not available";
    throw( std::runtime_error( s.str() ) );

  }

}

//-------------------------------------------------------------------------

int Numbers::RtHalf(const EBDetId& id) {

  int ic = id.ic();
  int ie = (ic-1)/20 + 1;
  int ip = (ic-1)%20 + 1;

  if( ie > 5 && ip < 11 ) return 1;

  return 0;

}

//-------------------------------------------------------------------------

int Numbers::RtHalf(const EEDetId& id) {

  int ix = id.ix();

  int ism = Numbers::iSM( id );

//
// to be confirmed !
//
  if ( ism == +5 && ix > 50 ) return 2;
  if ( ism == -5 && ix > 50 ) return 2;

  return 1;

}

//-------------------------------------------------------------------------

std::vector<DetId> Numbers::crystals( const EcalElectronicsId& id ) throw( std::runtime_error ) {

  if( Numbers::map ) {

    int idcc = id.dccId();
    int itt = id.towerId();

    return( Numbers::map->dccTowerConstituents( idcc, itt ) );

  } else {

    std::ostringstream s;
    s << "ECAL Geometry not available";
    throw( std::runtime_error( s.str() ) );

  }

}

//-------------------------------------------------------------------------

int Numbers::indexEB( const int ism, const int ie, const int ip ){

  return( (ip-1) + 20*(ie-1) + 1 );

}

//-------------------------------------------------------------------------

int Numbers::indexEE( const int ism, const int ix, const int iy ){

  int iz = 0;

  if( ism >=  1 && ism <=  9 ) iz = -1;
  if( ism >= 10 && ism <= 18 ) iz = +1;

  if( EEDetId::validDetId(ix, iy, iz) ) {

    return( 1000*ix + iy );

  } else {

    return( -1 );

  }

}

//-------------------------------------------------------------------------

int Numbers::icEB( const int ism, const int ie, const int ip ) {

  return( (ip-1) + 20*(ie-1) + 1 );

}

//-------------------------------------------------------------------------

int Numbers::icEE( const int ism, const int ix, const int iy ) throw( std::runtime_error ) {

  int iz = 0;

  if( ism >=  1 && ism <=  9 ) iz = -1;
  if( ism >= 10 && ism <= 18 ) iz = +1;

  if( EEDetId::validDetId(ix, iy, iz) ) {

    EEDetId id(ix, iy, iz, EEDetId::XYMODE);

    if( Numbers::iSM( id ) != ism ) return( -1 );

    if( Numbers::map ) {

      EcalElectronicsId eid = Numbers::map->getElectronicsId(id);

      int vfe = eid.towerId();
      int strip = eid.stripId();
      int channel = eid.xtalId();

      return ( (vfe-1)*25 + (strip-1)*5 + channel );

    } else {

      std::ostringstream s;
      s << "ECAL Geometry not available";
      throw( std::runtime_error( s.str() ) );

    }

  } else {

    return( -1 );

  }

}

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

int Numbers::ixSectorsEE[202] = {61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 0,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100,  0, 61, 65, 65, 70, 70, 80, 80, 90, 90, 92,  0, 61, 65, 65, 90, 90, 97,  0, 57, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80,  0, 50, 50,  0, 43, 40, 40, 35, 35, 30, 30, 25, 25, 20, 20,  0, 39, 35, 35, 10, 10,  3,  0, 39, 35, 35, 30, 30, 20, 20, 10, 10,  8,  0, 45, 45, 40, 40, 35, 35,  0, 55, 55, 60, 60, 65, 65};

int Numbers::iySectorsEE[202] = {50, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 50,  0, 50, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 50,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 50, 50, 55, 55, 60, 60,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 61,100,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 50, 50, 55, 55, 60, 60,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 39, 30, 30, 15, 15,  5,  0, 39, 30, 30, 15, 15,  5};

//-------------------------------------------------------------------------

// from EC-_DCC.gif
//int Numbers::inTowersEE[400] = { 0, 0, 0, 0, 0, 0, 0, 281, 293, 297, 148, 144, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 268, 278, 276, 280, 292, 296, 147, 143, 131, 127, 129, 119, 0, 0, 0, 0, 0, 0, 0, 268, 274, 277, 275, 279, 291, 295, 146, 142, 130, 126, 128, 125, 119, 0, 0, 0, 0, 0, 261, 273, 272, 271, 270, 269, 290, 294, 145, 141, 120, 121, 122, 123, 124, 112, 0, 0, 0, 261, 267, 266, 265, 264, 263, 262, 285, 289, 140, 136, 113, 114, 115, 116, 117, 118, 112, 0, 0, 254, 260, 259, 258, 257, 256, 255, 284, 288, 139, 135, 106, 107, 108, 109, 110, 111, 105, 0, 0, 247, 253, 252, 251, 250, 249, 248, 283, 287, 138, 134, 99, 100, 101, 102, 103, 104, 98, 0, 298, 246, 245, 244, 243, 242, 241, 240, 282, 286, 137, 133, 91, 92, 93, 94, 95, 96, 97, 149, 239, 238, 237, 236, 235, 234, 233, 232, 268, 261, 112, 119, 83, 84, 85, 86, 87, 88, 89, 90, 231, 230, 229, 228, 227, 226, 225, 224, 0, 0, 0, 0, 75, 76, 77, 78, 79, 80, 81, 82, 223, 222, 221, 220, 219, 218, 217, 216, 0, 0, 0, 0, 67, 68, 69, 70, 71, 72, 73, 74, 215, 214, 213, 212, 211, 210, 209, 208, 207, 178, 29, 58, 59, 60, 61, 62, 63, 64, 65, 66, 207, 206, 205, 204, 203, 202, 201, 200, 165, 161, 12, 16, 51, 52, 53, 54, 55, 56, 57, 58, 0, 199, 198, 197, 196, 195, 194, 193, 164, 160, 11, 15, 44, 45, 46, 47, 48, 49, 50, 0, 0, 192, 191, 190, 189, 188, 187, 186, 163, 159, 10, 14, 37, 38, 39, 40, 41, 42, 43, 0, 0, 207, 185, 184, 183, 181, 180, 179, 162, 158, 9, 13, 30, 31, 32, 34, 35, 36, 58, 0, 0, 0, 178, 177, 176, 175, 174, 173, 157, 153, 4, 8, 24, 25, 26, 27, 28, 29, 0, 0, 0, 0, 0, 178, 172, 171, 170, 169, 156, 152, 3, 7, 20, 21, 22, 23, 29, 0, 0, 0, 0, 0, 0, 0, 182, 168, 167, 166, 155, 151, 2, 6, 17, 18, 19, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 182, 154, 150, 1, 5, 33, 0, 0, 0, 0, 0, 0, 0};

// from EC-_DCC_Chan_from_IP.gif
int Numbers::inTowersEE[400] = { 0, 0, 0, 0, 0, 0, 0, 3, 13, 17, 34, 30, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 31, 29, 2, 12, 16, 33, 29, 19, 29, 31, 21, 0, 0, 0, 0, 0, 0, 0, 21, 27, 30, 28, 1, 11, 15, 32, 28, 18, 28, 30, 27, 21, 0, 0, 0, 0, 0, 14, 26, 25, 24, 23, 22, 10, 14, 31, 27, 22, 23, 24, 25, 26, 14, 0, 0, 0, 14, 20, 19, 18, 17, 16, 15, 5, 9, 26, 22, 15, 16, 17, 18, 19, 20, 14, 0, 0, 33, 13, 12, 11, 10, 9, 8, 4, 8, 25, 21, 8, 9, 10, 11, 12, 13, 33, 0, 0, 30, 32, 31, 7, 6, 5, 4, 33, 7, 24, 33, 4, 5, 6, 7, 31, 32, 30, 0, 34, 29, 28, 27, 26, 25, 3, 2, 32, 6, 23, 32, 2, 3, 25, 26, 27, 28, 29, 34, 24, 23, 22, 21, 20, 19, 18, 1, 21, 14, 21, 14, 1, 18, 19, 20, 21, 22, 23, 24, 17, 16, 15, 14, 13, 12, 11, 10, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 16, 17, 9, 8, 7, 6, 5, 4, 3, 32, 0, 0, 0, 0, 32, 3, 4, 5, 6, 7, 8, 9, 2, 1, 31, 30, 29, 28, 27, 26, 25, 3, 25, 3, 26, 27, 28, 29, 30, 31, 1, 2, 25, 24, 23, 22, 21, 20, 19, 18, 16, 12, 12, 16, 18, 19, 20, 21, 22, 23, 24, 25, 0, 17, 16, 15, 14, 13, 12, 33, 15, 11, 11, 15, 33, 12, 13, 14, 15, 16, 17, 0, 0, 11, 10, 9, 8, 7, 32, 31, 14, 10, 10, 14, 31, 32, 7, 8, 9, 10, 11, 0, 0, 25, 6, 5, 4, 29, 28, 27, 13, 9, 9, 13, 27, 28, 29, 4, 5, 6, 25, 0, 0, 0, 3, 2, 1, 26, 25, 24, 8, 4, 4, 8, 24, 25, 26, 1, 2, 3, 0, 0, 0, 0, 0, 3, 23, 22, 21, 20, 7, 3, 3, 7, 20, 21, 22, 23, 3, 0, 0, 0, 0, 0, 0, 0, 30, 19, 18, 17, 6, 2, 2, 6, 17, 18, 19, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 5, 1, 1, 5, 30, 0, 0, 0, 0, 0, 0, 0};

//-------------------------------------------------------------------------

int Numbers::ix0EE( const int ism ) {

  if( ism == 1 || ism == 15 ) return( -  5 );
  if( ism == 2 || ism == 14 ) return( +  0 );
  if( ism == 3 || ism == 13 ) return( + 10 );
  if( ism == 4 || ism == 12 ) return( + 40 );
  if( ism == 5 || ism == 11 ) return( + 50 );
  if( ism == 6 || ism == 10 ) return( + 55 );
  if( ism == 7 || ism == 18 ) return( + 50 );
  if( ism == 8 || ism == 17 ) return( + 25 );
  if( ism == 9 || ism == 16 ) return( +  0 );

  return( + 0 );

}

//-------------------------------------------------------------------------

int Numbers::iy0EE( const int ism ) {

  if( ism == 1 || ism == 10 ) return( + 20 );
  if( ism == 2 || ism == 11 ) return( + 45 );
  if( ism == 3 || ism == 12 ) return( + 55 );
  if( ism == 4 || ism == 13 ) return( + 55 );
  if( ism == 5 || ism == 14 ) return( + 45 );
  if( ism == 6 || ism == 15 ) return( + 20 );
  if( ism == 7 || ism == 16 ) return( +  0 );
  if( ism == 8 || ism == 17 ) return( -  5 );
  if( ism == 9 || ism == 18 ) return( +  0 );

  return( + 0 );

}

//-------------------------------------------------------------------------

bool Numbers::validEE( const int ism, const int ix, const int iy ) {

  int iz = 0;

  if( ism >=  1 && ism <=  9 ) iz = -1;
  if( ism >= 10 && ism <= 18 ) iz = +1;

  if( EEDetId::validDetId(ix, iy, iz) ) {

    EEDetId id(ix, iy, iz, EEDetId::XYMODE);

    if( Numbers::iSM( id ) == ism ) return true;

  }

  return false;

}

//-------------------------------------------------------------------------
