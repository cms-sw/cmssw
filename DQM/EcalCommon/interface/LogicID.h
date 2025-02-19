#ifndef LogicID_H
#define LogicID_H

/*!
  \file LogicID.h
  \brief Cache logicID vector from database
  \author B. Gobbo 
  \version $Revision: 1.15 $
  \date $Date: 2012/04/26 07:44:39 $
*/

#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LogicID {

public:

//-------------------------------------------------------------------------

// WARNING:
// this file depends on the content of
// OnlineDB/EcalCondDB/perl/lib/CondDB/channelView.pm

//-------------------------------------------------------------------------

static EcalLogicID getEcalLogicID( const char* name,
                                   const int id1 = EcalLogicID::NULLID,
                                   const int id2 = EcalLogicID::NULLID,
                                   const int id3 = EcalLogicID::NULLID ) throw( std::runtime_error ) {

  // EcalBarrel

  if( strcmp(name, "EB") == 0 ) {
    return( EcalLogicID( "EB",
                         1000000000UL ) );
  }
  if( strcmp(name, "EB_crystal_number") == 0 ) {
    return( EcalLogicID( "EB_crystal_number",
                         1011000000UL+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( strcmp(name, "EB_trigger_tower") == 0 ) {
    return( EcalLogicID( "EB_trigger_tower",
                         1021000000UL+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( strcmp(name, "EB_mem_channel") == 0 ) {
    return( EcalLogicID( "EB_mem_channel",
                         1191000000UL+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( strcmp(name, "EB_mem_TT") == 0 ) {
    return( EcalLogicID( "EB_mem_TT",
                         1181000000UL+10000*id1+id2,
                         id1,
                         id2 ) );
  }
  if( strcmp(name, "EB_LM_PN") == 0 ) {
    return( EcalLogicID( "EB_LM_PN",
                         1131000000UL+10000*id1+id2,
                         id1,
                         id2 ) );
  }

  // EcalEndcap

  if( strcmp(name, "EE") == 0 ) {
    return( EcalLogicID( "EE",
                         2000000000UL ) );
  }
  if( strcmp(name, "EE_crystal_number") == 0 ) {
    return( EcalLogicID( "EE_crystal_number",
                         2010000000UL+1000000*((id1>=1&&id1<=9)?2:0)+1000*int(id2/1000)+int(id2%1000),
                         (id1>=1&&id1<=9)?+1:-1,
                         int(id2/1000),
                         int(id2%1000) ) );
  }
  if( strcmp(name, "EE_readout_tower") == 0 ) {
    return( EcalLogicID( "EE_readout_tower",
                         2110000000UL+100*((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-10)))+id2,
                         ((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-10))),
                         id2 ) );
  }
  if( strcmp(name, "EE_mem_channel") == 0 ) {
    return( EcalLogicID( "EE_mem_channel",
                         100*((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-10)))+id2,
                         ((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-10))),
                         id2 ) );
  }
  if( strcmp(name, "EE_mem_TT") == 0 ) {
    return( EcalLogicID( "EE_mem_TT",
                         100*((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-10)))+id2,
                         ((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-10))),
                         id2 ) );
  }
  if( strcmp(name, "EE_LM_PN") == 0 ) {
    return( EcalLogicID( "EE_LM_PN",
                         100*((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-10)))+id2,
                         ((id1>=1&&id1<=9)?(646+(id1-1)):(601+(id1-10))),
                         id2 ) );
  }

  throw( std::runtime_error( "Unknown 'name': " + std::string( name ) ) );
  return( EcalLogicID( std::string( "" ), EcalLogicID::NULLID ) );

}

private:

  LogicID() {}; // Hidden to force static use
  ~LogicID() {}; // Hidden to force static use

};

#endif // LogicID_H
