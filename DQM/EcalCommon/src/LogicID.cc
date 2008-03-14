// $Id: LogicID.cc,v 1.9 2008/02/09 19:50:10 dellaric Exp $

/*!
  \file LogicID.cc
  \brief Construct EcalLogicIDs
  \author G. Della Ricca
  \author B. Gobbo
  \version $Revision: 1.9 $
  \date $Date: 2008/02/09 19:50:10 $
*/

#include "DQM/EcalCommon/interface/LogicID.h"

//-------------------------------------------------------------------------

// WARNING:
// this file depends on the content of
// OnlineDB/EcalCondDB/perl/lib/CondDB/channelView.pm

//-------------------------------------------------------------------------

EcalLogicID LogicID::getEcalLogicID( std::string name,
                                     int id1,
                                     int id2,
                                     int id3 ) throw( std::runtime_error ) {

  // EcalBarrel

  if( name == "EB" ) {
    return( EcalLogicID( "EB",
                         1000000000 ) );
  }
  if( name == "EB_crystal_number" ) {
    return( EcalLogicID( "EB_crystal_number",
                         1011000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( name == "EB_trigger_tower" ) {
    return( EcalLogicID( "EB_trigger_tower",
                         1021000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( name == "EB_mem_channel" ) {
    return( EcalLogicID( "EB_mem_channel",
                         1191000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( name == "EB_mem_TT" ) {
    return( EcalLogicID( "EB_mem_TT",
                         1181000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( name == "EB_LM_PN" ) {
    return( EcalLogicID( "EB_LM_PN",
                         1131000000+10000*id1+id2,
                         id1,
                         id2 ) );
  }

  // EcalEndcap

  if( name == "EE" ) {
    return( EcalLogicID( "EE",
                         2000000000 ) );
  }
  if( name == "EE_crystal_number" ) {
    return( EcalLogicID( "EE_crystal_number",
                         2010000000+1000000*((id1>=1&&id1<=9)?2:0)+1000*int(id2/1000)+int(id2%1000),
                         (id1>=1&&id1<=9)?+1:-1,
                         int(id2/1000),
                         int(id2%1000) ) );
  }
  if( name == "EE_readout_tower" ) {
    return( EcalLogicID( "EE_readout_tower",
                         2000000000+100*((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-9)))+id2,
                         ((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-9))),
                         id2 ) );
  }
  if( name == "EE_mem_channel" ) {
    return( EcalLogicID( "EE_mem_channel", EcalLogicID::NULLID ) );
  }
  if( name == "EE_mem_TT" ) {
    return( EcalLogicID( "EE_mem_TT", EcalLogicID::NULLID ) );
  }
  if( name == "EE_LM_PN" ) {
    return( EcalLogicID( "EE_mem_TT", EcalLogicID::NULLID ) );
  }

  throw( std::runtime_error( "Unknown 'name': " + name ) );
  return( EcalLogicID( std::string( "" ), EcalLogicID::NULLID ) );

}

