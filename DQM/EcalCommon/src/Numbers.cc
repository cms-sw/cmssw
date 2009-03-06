// $Id: Numbers.cc,v 1.62 2008/08/13 13:22:04 dellaric Exp $

/*!
  \file Numbers.cc
  \brief Some "id" conversions
  \author B. Gobbo
  \version $Revision: 1.62 $
  \date $Date: 2008/08/13 13:22:04 $
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

//-------------------------------------------------------------------------

void Numbers::initGeometry( const edm::EventSetup& setup, bool verbose ) {

  if( Numbers::init ) return;

  if ( verbose ) std::cout << "Initializing EcalElectronicsMapping ..." << std::endl;

  Numbers::init = true;

  edm::ESHandle< EcalElectronicsMapping > handle;
  setup.get< EcalMappingRcd >().get(handle);
  Numbers::map = handle.product();

  if ( verbose ) std::cout << "done." << std::endl;

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

  if( Numbers::map ) {

    EcalElectronicsId eid = Numbers::map->getElectronicsId(id);
    int idcc = eid.dccId();

    // EB-/EB+
    if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

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

    if( Numbers::map ) {

      int idcc = Numbers::map->DCCid(id);

      // EB-/EB+
      if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

      std::ostringstream s;
      s << "Wrong DCC id: dcc = " << idcc;
      throw( std::runtime_error( s.str() ) );

    } else {

      std::ostringstream s;
      s << "ECAL Geometry not available";
      throw( std::runtime_error( s.str() ) );

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

    if( Numbers::map ) {

      return( Numbers::map->iTT(id) );

    } else {

      std::ostringstream s;
      s << "ECAL Geometry not available";
      throw( std::runtime_error( s.str() ) );

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

  // EE-05
  if ( ism ==  8 && ix > 50 ) return 1;

  // EE+05
  if ( ism == 17 && ix > 50 ) return 1;

  return 0;

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

      // EE-05 & EE+05
      if( ism == 8 || ism == 17 ) {
        if( vfe > 17 ) vfe = vfe - 7;
      }

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

int Numbers::ixSectorsEE[202] = {61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 0,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100,  0, 61, 65, 65, 70, 70, 80, 80, 90, 90, 92,  0, 61, 65, 65, 90, 90, 97,  0, 57, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80,  0, 50, 50,  0, 43, 40, 40, 35, 35, 30, 30, 25, 25, 20, 20,  0, 39, 35, 35, 10, 10,  3,  0, 39, 35, 35, 30, 30, 20, 20, 10, 10,  8,  0, 45, 45, 40, 40, 35, 35,  0, 55, 55, 60, 60, 65, 65};

int Numbers::iySectorsEE[202] = {50, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 50,  0, 50, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 50,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 50, 50, 55, 55, 60, 60,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 61,100,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 50, 50, 55, 55, 60, 60,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 39, 30, 30, 15, 15,  5,  0, 39, 30, 30, 15, 15,  5};

//-------------------------------------------------------------------------

// from EC-_CCUID_to_IP.gif

int Numbers::inTowersEE[400] = { 0, 0, 0, 0, 0, 0, 0, 27, 37, 41, 17, 13, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 31, 29, 26, 36, 40, 16, 12, 2, 29, 31, 21, 0, 0, 0, 0, 0, 0, 0, 21, 27, 30, 28, 25, 35, 39, 15, 11, 1, 28, 30, 27, 21, 0, 0, 0, 0, 0, 14, 26, 25, 24, 23, 22, 34, 38, 14, 10, 22, 23, 24, 25, 26, 14, 0, 0, 0, 14, 20, 19, 18, 17, 16, 15, 29, 33, 9, 5, 15, 16, 17, 18, 19, 20, 14, 0, 0, 33, 13, 12, 11, 10, 9, 8, 28, 32, 8, 4, 8, 9, 10, 11, 12, 13, 33, 0, 0, 30, 32, 31, 7, 6, 5, 4, 33, 31, 7, 33, 4, 5, 6, 7, 31, 32, 30, 0, 34, 29, 28, 27, 26, 25, 3, 2, 32, 30, 6, 32, 2, 3, 25, 26, 27, 28, 29, 34, 24, 23, 22, 21, 20, 19, 18, 1, 21, 14, 21, 14, 1, 18, 19, 20, 21, 22, 23, 24, 17, 16, 15, 14, 13, 12, 11, 10, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 16, 17, 9, 8, 7, 6, 5, 4, 3, 32, 0, 0, 0, 0, 32, 3, 4, 5, 6, 7, 8, 9, 2, 1, 31, 30, 29, 28, 27, 26, 25, 3, 25, 3, 26, 27, 28, 29, 30, 31, 1, 2, 25, 24, 23, 22, 21, 20, 19, 18, 16, 12, 12, 16, 18, 19, 20, 21, 22, 23, 24, 25, 0, 17, 16, 15, 14, 13, 12, 33, 15, 11, 11, 15, 33, 12, 13, 14, 15, 16, 17, 0, 0, 11, 10, 9, 8, 7, 32, 31, 14, 10, 10, 14, 31, 32, 7, 8, 9, 10, 11, 0, 0, 25, 6, 5, 4, 29, 28, 27, 13, 9, 9, 13, 27, 28, 29, 4, 5, 6, 25, 0, 0, 0, 3, 2, 1, 26, 25, 24, 8, 4, 4, 8, 24, 25, 26, 1, 2, 3, 0, 0, 0, 0, 0, 3, 23, 22, 21, 20, 7, 3, 3, 7, 20, 21, 22, 23, 3, 0, 0, 0, 0, 0, 0, 0, 30, 19, 18, 17, 6, 2, 2, 6, 17, 18, 19, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 5, 1, 1, 5, 30, 0, 0, 0, 0, 0, 0, 0};

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
