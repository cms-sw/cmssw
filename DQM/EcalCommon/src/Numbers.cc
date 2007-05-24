// $Id: Numbers.cc,v 1.9 2007/05/24 13:04:56 benigno Exp $

/*!
  \file Numbers.cc
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.9 $
  \date $Date: 2007/05/24 13:04:56 $
*/

#include <sstream>
#include <iomanip>
#include "DQM/EcalCommon/interface/Numbers.h"

//-------------------------------------------------------------------------
// To be removed in a (short) future... 
// Initialized for Barrel, Endcaps should set it by themselves
int Numbers::maxSM = 36;

//-------------------------------------------------------------------------

int Numbers::iEB( int ism ) throw( std::runtime_error ) {
  
  if( ism < 1 || ism > 36 ) {
    std::ostringstream s;
    s << "Wrong SM id determination: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );
    return( -999 );
  }

  return( ism < 19 ? -ism : ism - 18 );

}

//-------------------------------------------------------------------------

std::string Numbers::sEB( int ism  ) throw( std::runtime_error ) {

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

int Numbers::iEE( int ism ) throw( std::runtime_error ) {
  
  if( ism < 1 || ism > 18 ) {
    std::ostringstream s;
    s << "Wrong SM id determination: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );
    return( -999 );
  }

  return( ism < 10 ? -ism : ism - 9 );

}

//-------------------------------------------------------------------------

std::string Numbers::sEE( int ism  ) throw( std::runtime_error ) {

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

int Numbers::iSM( int ism ) throw( std::runtime_error ) {
  if( ism < 1 || ism > 36 ) {
    std::ostringstream s;
    s << "Wrong SM id determination: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );
    return( -999 );
  }

// To be removed in a (short) future... 
  if( ism > Numbers::maxSM ) {
    return (Numbers::maxSM+1);
  }

  return( ism < (Numbers::maxSM/2+1) ? ism+Numbers::maxSM/2 : ism-Numbers::maxSM/2 ); 
}

//-------------------------------------------------------------------------

int Numbers::iSM( const EBDetId& id ) {
  return( Numbers::iSM( id.ism() ) );
}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalTrigTowerDetId& id ) {
  return( Numbers::iSM( id.iDCC() ) );
}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalElectronicsId& id ) {
  return( Numbers::iSM( id.dccId() ) );
}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalPnDiodeDetId& id ) {
  return( Numbers::iSM( id.iDCCId() ) );
}

//-------------------------------------------------------------------------

int Numbers::iSM( const EcalDCCHeaderBlock& id ) {
  return( Numbers::iSM( id.id() ) );
}
