// $Id: LogicID.cc,v 1.14 2008/03/16 10:35:17 dellaric Exp $

/*!
  \file LogicID.cc
  \brief Construct EcalLogicIDs
  \author G. Della Ricca
  \author B. Gobbo
  \version $Revision: 1.14 $
  \date $Date: 2008/03/16 10:35:17 $
*/

#include "DQM/EcalCommon/interface/LogicID.h"

//-------------------------------------------------------------------------

// WARNING:
// this file depends on the content of
// OnlineDB/EcalCondDB/perl/lib/CondDB/channelView.pm

//-------------------------------------------------------------------------

EcalLogicID LogicID::getEcalLogicID( const char* name,
                                     const int id1,
                                     const int id2,
                                     const int id3 ) throw( std::runtime_error ) {

  // EcalBarrel

  if( strcmp(name, "EB") == 0 ) {
    return( EcalLogicID( "EB",
                         1000000000 ) );
  }
  if( strcmp(name, "EB_crystal_number") == 0 ) {
    return( EcalLogicID( "EB_crystal_number",
                         1011000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( strcmp(name, "EB_trigger_tower") == 0 ) {
    return( EcalLogicID( "EB_trigger_tower",
                         1021000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( strcmp(name, "EB_mem_channel") == 0 ) {
    return( EcalLogicID( "EB_mem_channel",
                         1191000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( strcmp(name, "EB_mem_TT") == 0 ) {
    return( EcalLogicID( "EB_mem_TT",
                         1181000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( strcmp(name, "EB_LM_PN") == 0 ) {
    return( EcalLogicID( "EB_LM_PN",
                         1131000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }

  // EcalEndcap

  if( strcmp(name, "EE") == 0 ) {
    return( EcalLogicID( "EE",
                         2000000000 ) );
  }
  if( strcmp(name, "EE_crystal_number") == 0 ) {
    return( EcalLogicID( "EE_crystal_number",
                         2010000000+1000000*((id1>=1&&id1<=9)?2:0)+1000*int(id2/1000)+int(id2%1000),
                         (id1>=1&&id1<=9)?+1:-1,
                         int(id2/1000),
                         int(id2%1000) ) );
  }
  if( strcmp(name, "EE_readout_tower") == 0 ) {
    return( EcalLogicID( "EE_readout_tower",
                         2000000000+100*((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-9)))+id2,
                         ((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-9))),
                         id2 ) );
  }
  if( strcmp(name, "EE_mem_channel") == 0 ) {
    return( EcalLogicID( "EE_mem_channel", EcalLogicID::NULLID ) );
  }
  if( strcmp(name, "EE_mem_TT") == 0 ) {
    return( EcalLogicID( "EE_mem_TT", EcalLogicID::NULLID ) );
  }
  if( strcmp(name, "EE_LM_PN") == 0 ) {
    return( EcalLogicID( "EE_mem_TT", EcalLogicID::NULLID ) );
  }

  throw( std::runtime_error( "Unknown 'name': " + std::string( name ) ) );
  return( EcalLogicID( std::string( "" ), EcalLogicID::NULLID ) );

}

