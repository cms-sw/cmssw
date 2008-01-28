// $Id: LogicID.cc,v 1.7 2008/01/28 13:04:23 dellaric Exp $

/*!
  \file LogicID.cc
  \brief Cache logicID vector from database
  \author B. Gobbo 
  \version $Revision: 1.7 $
  \date $Date: 2008/01/28 13:04:23 $
*/

#include <sstream>
#include <iomanip>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/LogicID.h"

bool                                              LogicID::init_ = false;
std::map< std::string, std::vector<EcalLogicID> > LogicID::IDmap_;

//-------------------------------------------------------------------------

void LogicID::init( EcalCondDBInterface* eConn, EcalSubdetector subdet ) {

  if ( subdet == EcalBarrel ) {

    IDmap_[ "EB_crystal_number" ] = eConn->getEcalLogicIDSet( "EB_crystal_number", 1, 36,  1, 1700 );
    IDmap_[ "EB_mem_channel" ]    = eConn->getEcalLogicIDSet( "EB_mem_channel",    1, 36,  1,   50 );
    IDmap_[ "EB_trigger_tower" ]  = eConn->getEcalLogicIDSet( "EB_trigger_tower",  1, 36,  1,   68 );
    IDmap_[ "EB_LM_PN" ]          = eConn->getEcalLogicIDSet( "EB_LM_PN",          1, 36,  0,    9 );
    IDmap_[ "EB_mem_TT" ]         = eConn->getEcalLogicIDSet( "EB_mem_TT",         1, 36, 69,   70 );
    IDmap_[ "EB" ].push_back( eConn->getEcalLogicID( "EB" ) );

  }
  else if ( subdet == EcalEndcap ) {

    IDmap_[ "EE" ].push_back( eConn->getEcalLogicID( "EE" ) );

  }
  else {
    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( std::runtime_error( s.str() ) );
    return;
  }

  init_ = true;

  return;

}

//-------------------------------------------------------------------------

EcalLogicID LogicID::getEcalLogicID( std::string name, int id1, int id2, int id3 ) throw( std::runtime_error ) {

  if( init_ ) {
    if( name == "EB_crystal_number" ) {
      return( IDmap_[ name ][ 1700*(id1-1)+id2-1 ] );
    }
    else if( name == "EB_mem_channel" ) {
      return( IDmap_[ name ][ 50*(id1-1)+id2-1 ] );
    }
    else if( name == "EB_trigger_tower" ) {
      return( IDmap_[ name ][ 68*(id1-1)+id2-1 ] );
    }
    else if( name == "EB_LM_PN" ) {
      return( IDmap_[ name ][ 10*(id1-1)+id2 ] );
    }
    else if( name == "EB_mem_TT" ) {
      return( IDmap_[ name ][ 2*(id1-1)+id2-69 ] );
    }
    else if( name == "EB" ) {
      return( IDmap_[ name ][ 0 ] );
    }
    else {
      throw( std::runtime_error( "Unknown 'name': " + name ) );
      return( EcalLogicID( std::string( "" ), EcalLogicID::NULLID ) );
    }
  }
  else {
    throw( std::runtime_error( "LogicID static object not yet initialized." ) );
    return ( EcalLogicID( std::string( "" ), EcalLogicID::NULLID ) );
  }

}
