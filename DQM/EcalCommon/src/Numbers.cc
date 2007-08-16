// $Id: Numbers.cc,v 1.14 2007/08/16 14:26:07 dellaric Exp $

/*!
  \file Numbers.cc
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.14 $
  \date $Date: 2007/08/16 14:26:07 $
*/

#include <sstream>
#include <iomanip>
#include "DQM/EcalCommon/interface/Numbers.h"

//-------------------------------------------------------------------------

const EcalElectronicsMapping* Numbers::map = 0;

bool Numbers::init = false;

//-------------------------------------------------------------------------

void Numbers::initGeometry( const edm::EventSetup& setup ) {

  if ( Numbers::init ) return;

  std::cout << "Initializing ECAL Geometry ... " << std::flush;

  Numbers::init = true;

  try {
    edm::ESHandle< EcalElectronicsMapping > handle;
    setup.get< EcalMappingRcd >().get(handle);
    Numbers::map = handle.product();
    std::cout << "done." << std::endl;
  } catch (cms::Exception &e) {
    std::cout << "not available" << std::endl;
  }
  std::cout << std::endl;

}

//-------------------------------------------------------------------------

int Numbers::iEB( const int ism ) throw( std::runtime_error ) {
  
  if( ism < 1 || ism > 36 ) {
    std::ostringstream s;
    s << "Wrong SM id determination: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );
    return( -999 );
  }

  int ieb = - 999;

  if( ism >=  1 && ism <= 18 ) ieb = -ism;
  if( ism >= 19 && ism <= 36 ) ieb = +ism - 18;

  return( ieb );

}

//-------------------------------------------------------------------------

std::string Numbers::sEB( const int ism  ) throw( std::runtime_error ) {

  try {
    int ieb = Numbers::iEB( ism );
    std::ostringstream s;
    s << "EB" << std::setw(3) << std::setfill('0')
      << std::setiosflags( std::ios::showpos )
      << std::setiosflags( std::ios::internal )
      << ieb
      << std::resetiosflags( std::ios::showpos )
      << std::resetiosflags( std::ios::internal )
      << std::ends;
    return( s.str() );
  } catch( std::runtime_error &e ) {
    throw( std::runtime_error( e.what() ) );
    return( "" );
  }
  
}

//-------------------------------------------------------------------------

int Numbers::iEE( const int ism ) throw( std::runtime_error ) {
  
  if( ism < 1 || ism > 18 ) {
    std::ostringstream s;
    s << "Wrong SM id determination: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );
    return( -999 );
  }

  int iee = -999;

  if( ism ==  1 ) iee = -7;
  if( ism ==  2 ) iee = -8;
  if( ism ==  3 ) iee = -9;
  if( ism ==  4 ) iee = -1;
  if( ism ==  5 ) iee = -2;
  if( ism ==  6 ) iee = -3;
  if( ism ==  7 ) iee = -4;
  if( ism ==  8 ) iee = -5;
  if( ism ==  9 ) iee = -6;
  if( ism == 10 ) iee = +7;
  if( ism == 11 ) iee = +8;
  if( ism == 12 ) iee = +9;
  if( ism == 13 ) iee = +1;
  if( ism == 14 ) iee = +2;
  if( ism == 15 ) iee = +3;
  if( ism == 16 ) iee = +4;
  if( ism == 17 ) iee = +5;
  if( ism == 18 ) iee = +6;

  return( iee );

}

//-------------------------------------------------------------------------

std::string Numbers::sEE( const int ism  ) throw( std::runtime_error ) {

  try {
    int iee = Numbers::iEE( ism );
    std::ostringstream s;
    s << "EE" << std::setw(3) << std::setfill('0')
      << std::setiosflags( std::ios::showpos )
      << std::setiosflags( std::ios::internal )
      << iee
      << std::resetiosflags( std::ios::showpos )
      << std::resetiosflags( std::ios::internal )
      << std::ends;
    return( s.str() );
  } catch( std::runtime_error &e ) {
    throw( std::runtime_error( e.what() ) );
    return( "" );
  }
  
}

//-------------------------------------------------------------------------

int Numbers::iSM( const int ism, const int subdet ) throw( std::runtime_error ) {

  if( subdet == EcalBarrel ) {
    if( ism < 1 || ism > 36 ) {
      std::ostringstream s;
      s << "Wrong SM id: iSM = " << ism;
      throw( std::runtime_error( s.str() ) );
      return( -999 );
    }
    if( ism <= 18 ) {
      return( ism+18 );
    } else {
      return( ism-18 );
    }
  } else if( subdet ==  EcalEndcap) {
    if( ism < 1 || ism > 18 ) {
      std::ostringstream s;
      s << "Wrong SM id: iSM = " << ism;
      throw( std::runtime_error( s.str() ) );
      return( -999 );
    }
    if( ism <= 9 ) {
      return( ism+9 );
    } else {
      return( ism-9 );
    }
  } else {
    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( std::runtime_error( s.str() ) );
    return( -999 );
  }

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EBDetId& id ) throw( std::runtime_error ) {

  int ism = -999;

  if( Numbers::map ) {
    EcalElectronicsId eid = Numbers::map->getElectronicsId(id);
    int idcc = eid.dccId();
    if( idcc >= 10 && idcc <= 45 ) ism = idcc - 9;
  } else {
    ism = Numbers::iSM( id.ism(), EcalBarrel );
  }

  return( ism );

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EEDetId& id ) throw( std::runtime_error ) {

  int ism = -999;

  if( Numbers::map ) {
    EcalElectronicsId eid = Numbers::map->getElectronicsId(id);
    int idcc = eid.dccId();
    if( idcc >=  1 && idcc <=  9 ) ism = idcc;
    if( idcc >= 46 && idcc <= 54 ) ism = idcc - 45 + 9;
  } else {
    std::ostringstream s;
    s << "ECAL Geometry not available";
    throw( std::runtime_error( s.str() ) );
  }

  return( ism );

}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalTrigTowerDetId& id ) {
  return( Numbers::iSM( id.iDCC(), id.subDet() ) );
}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalElectronicsId& id ) {
  return( Numbers::iSM( id.dccId(), id.subdet() ) );
}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalPnDiodeDetId& id ) {
  return( Numbers::iSM( id.iDCCId(), id.iEcalSubDetectorId() ) );
}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalDCCHeaderBlock& id, const int subdet ) {

  // special case for testbeam/cosmic data
  if ( id.id() == 1 && subdet == EcalBarrel ) return( Numbers::iSM( id.id(), EcalBarrel ) );

  if ( id.id() >=  1 && id.id() <=  9 ) return( Numbers::iSM( id.id(), EcalEndcap ) );
  if ( id.id() >= 10 && id.id() <= 45 ) return( Numbers::iSM( id.id()-9, EcalBarrel ) );
  if ( id.id() >= 46 && id.id() <= 54 ) return( Numbers::iSM( id.id()-45+9, EcalEndcap ) );

  return( -999 );

}

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

int Numbers::ixSectorsEE[202] = {61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 0,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100,  0, 61, 65, 65, 70, 70, 80, 80, 90, 90, 92,  0, 61, 65, 65, 90, 90, 97,  0, 57, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80,  0, 50, 50,  0, 43, 40, 40, 35, 35, 30, 30, 25, 25, 20, 20,  0, 39, 35, 35, 10, 10,  3,  0, 39, 35, 35, 30, 30, 20, 20, 10, 10,  8,  0, 45, 45, 40, 40, 35, 35,  0, 55, 55, 60, 60, 65, 65};

int Numbers::iySectorsEE[202] = {50, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 50,  0, 50, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 50,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 50, 50, 55, 55, 60, 60,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 61,100,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 50, 50, 55, 55, 60, 60,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 39, 30, 30, 15, 15,  5,  0, 39, 30, 30, 15, 15,  5};

//-------------------------------------------------------------------------

int Numbers::ix0EE( const int ism ) {

  int ix = 0;

  if ( ism == 1 || ism == 10 ) ix = -  5;
  if ( ism == 2 || ism == 11 ) ix = +  0;
  if ( ism == 3 || ism == 12 ) ix = + 10;
  if ( ism == 4 || ism == 13 ) ix = + 40;
  if ( ism == 5 || ism == 14 ) ix = + 50;
  if ( ism == 6 || ism == 15 ) ix = + 55;
  if ( ism == 7 || ism == 16 ) ix = + 50;
  if ( ism == 8 || ism == 17 ) ix = + 25;
  if ( ism == 9 || ism == 18 ) ix = +  0;

  return ix;

}

//-------------------------------------------------------------------------

int Numbers::iy0EE( const int ism ) {

  int iy = 0;

  if ( ism == 1 || ism == 10 ) iy = + 20;
  if ( ism == 2 || ism == 11 ) iy = + 45;
  if ( ism == 3 || ism == 12 ) iy = + 55; 
  if ( ism == 4 || ism == 13 ) iy = + 55; 
  if ( ism == 5 || ism == 14 ) iy = + 45; 
  if ( ism == 6 || ism == 15 ) iy = + 20;
  if ( ism == 7 || ism == 16 ) iy = +  0;
  if ( ism == 8 || ism == 17 ) iy = -  5;
  if ( ism == 9 || ism == 18 ) iy = +  0;

  return iy;

}

//-------------------------------------------------------------------------

bool Numbers::validEE( const int ism, const int ix, const int iy ) {

  EEDetId id0;

  int iz = 0;

  if ( ism >=  1 && ism <=  9 ) iz = -1;
  if ( ism >= 10 && ism <= 18 ) iz = +1;

  if ( id0.validDetId(ix, iy, iz) ) {

    EEDetId id1(ix, iy, iz, EEDetId::XYMODE);

    if ( Numbers::iSM( id1 ) == ism ) return true;

  }

  return false;

}

//-------------------------------------------------------------------------

int Numbers::icEE( const int ism, const int ix, const int iy ) {

  EEDetId id0;

  int iz = 0;

  if ( ism >=  1 && ism <=  9 ) iz = -1;
  if ( ism >= 10 && ism <= 18 ) iz = +1;

  if ( id0.validDetId(ix, iy, iz) ) {

    EEDetId id1(ix, iy, iz, EEDetId::XYMODE);

    //if ( Numbers::iSM( id1 ) == ism ) return( id1.ic() );

   // temporary fix, waiting for something better ....

    return( 5*(ix-1) + (iy-1) );

  }

  return( -1 );

}

//-------------------------------------------------------------------------
