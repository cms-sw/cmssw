// $Id: $

/*!
  \file Numbers.cc
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: $
  \date $Date: $
*/

#include <sstream>
#include <iomanip>
#include "DQM/EcalCommon/interface/Numbers.h"

//-------------------------------------------------------------------------

int Numbers::iEB( int fed ) throw( std::runtime_error ) {
  
  int ism = 0;
  if( fed > 600 ) {
    ism = fed - 609;
  }
  else {
    ism = fed -9;
  }

  if( ism < 1 || ism > 36 ) {
    std::ostringstream s;
    s << "Wrong SM id determination: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );
    return( -999 );
  }

  return( ism < 19 ? -ism : ism - 18 );

}

//-------------------------------------------------------------------------

std::string Numbers::sEB( int fed  ) throw( std::runtime_error ) {

  try {
    int ieb = Numbers::iEB( fed );
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
