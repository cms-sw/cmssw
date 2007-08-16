/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/08/16 10:53:27 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------
// Constructors --
//----------------
DTDeadFlag::DTDeadFlag():
  dataVersion( " " ) {
}


DTDeadFlag::DTDeadFlag( const std::string& version ):
  dataVersion( version ) {
}


DTDeadFlagId::DTDeadFlagId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ) {
}


DTDeadFlagData::DTDeadFlagData() :
   dead_HV( false ),
   dead_TP( false ),
   dead_RO( false ),
   discCat( false ) {
}


//--------------
// Destructor --
//--------------
DTDeadFlag::~DTDeadFlag() {
}


DTDeadFlagId::~DTDeadFlagId() {
}


DTDeadFlagData::~DTDeadFlagData() {
}


//--------------
// Operations --
//--------------
bool DTDeadFlagCompare::operator()( const DTDeadFlagId& idl,
                                      const DTDeadFlagId& idr ) const {
  if ( idl.  wheelId < idr.  wheelId ) return true;
  if ( idl.  wheelId > idr.  wheelId ) return false;
  if ( idl.stationId < idr.stationId ) return true;
  if ( idl.stationId > idr.stationId ) return false;
  if ( idl. sectorId < idr. sectorId ) return true;
  if ( idl. sectorId > idr. sectorId ) return false;
  if ( idl.     slId < idr.     slId ) return true;
  if ( idl.     slId > idr.     slId ) return false;
  if ( idl.  layerId < idr.  layerId ) return true;
  if ( idl.  layerId > idr.  layerId ) return false;
  if ( idl.   cellId < idr.   cellId ) return true;
  if ( idl.   cellId > idr.   cellId ) return false;
  return false;
}


int DTDeadFlag::cellStatus( int   wheelId,
                            int stationId,
                            int  sectorId,
                            int      slId,
                            int   layerId,
                            int    cellId,
                            bool& dead_HV,
                            bool& dead_TP,
                            bool& dead_RO,
                            bool& discCat ) const {

  dead_HV = false;
  dead_TP = false;
  dead_RO = false;
  discCat = false;

  DTDeadFlagId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  key.  layerId =   layerId;
  key.   cellId =    cellId;
  std::map<DTDeadFlagId,
           DTDeadFlagData,
           DTDeadFlagCompare>::const_iterator iter = cellData.find( key );

  if ( iter != cellData.end() ) {
    const DTDeadFlagData& data = iter->second;
    dead_HV = data.dead_HV;
    dead_TP = data.dead_TP;
    dead_RO = data.dead_RO;
    discCat = data.discCat;
    return 0;
  }
  return 1;

}


int DTDeadFlag::cellStatus( const DTWireId& id,
                            bool& dead_HV,
                            bool& dead_TP,
                            bool& dead_RO,
                            bool& discCat ) const {
  return cellStatus( id.wheel(),
                     id.station(),
                     id.sector(),
                     id.superLayer(),
                     id.layer(),
                     id.wire(),
                     dead_HV, dead_TP, dead_RO, discCat );
}


const
std::string& DTDeadFlag::version() const {
  return dataVersion;
}


std::string& DTDeadFlag::version() {
  return dataVersion;
}


void DTDeadFlag::clear() {
  cellData.clear();
  return;
}


int DTDeadFlag::setCellStatus( int   wheelId,
                               int stationId,
                               int  sectorId,
                               int      slId,
                               int   layerId,
                               int    cellId,
                               bool dead_HV,
                               bool dead_TP,
                               bool dead_RO,
                               bool discCat ) {

  DTDeadFlagId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  key.  layerId =   layerId;
  key.   cellId =    cellId;

  std::map<DTDeadFlagId,
           DTDeadFlagData,
           DTDeadFlagCompare>::iterator iter = cellData.find( key );
  if ( iter != cellData.end() ) {
    DTDeadFlagData& data = iter->second;
    data.dead_HV = dead_HV;
    data.dead_TP = dead_TP;
    data.dead_RO = dead_RO;
    data.discCat = discCat;
    return -1;
  }
  else {
    DTDeadFlagData data;
    data.dead_HV = dead_HV;
    data.dead_TP = dead_TP;
    data.dead_RO = dead_RO;
    data.discCat = discCat;
    cellData.insert( std::pair<const DTDeadFlagId,
                                     DTDeadFlagData>( key, data ) );
    return 0;
  }

  return 99;

}


int DTDeadFlag::setCellStatus( const DTWireId& id,
                               bool dead_HV,
                               bool dead_TP,
                               bool dead_RO,
                               bool discCat ) {
  return setCellStatus( id.wheel(),
                        id.station(),
                        id.sector(),
                        id.superLayer(),
                        id.layer(),
                        id.wire(),
                        dead_HV, dead_TP, dead_RO, discCat );
}


int DTDeadFlag::setCellDead_HV( int   wheelId,
                                int stationId,
                                int  sectorId,
                                int      slId,
                                int   layerId,
                                int    cellId,
                                bool flag ) {

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                             dead_HV, dead_TP, dead_RO, discCat );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                      flag, dead_TP, dead_RO, discCat );
  return status;

}


int DTDeadFlag::setCellDead_HV( const DTWireId& id,
                                bool flag ) {
  return setCellDead_HV( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire(),
                         flag );
}


int DTDeadFlag::setCellDead_TP( int   wheelId,
                                int stationId,
                                int  sectorId,
                                int      slId,
                                int   layerId,
                                int    cellId,
                                bool flag ) {

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                             dead_HV, dead_TP, dead_RO, discCat );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                   dead_HV, flag, dead_RO, discCat );
  return status;

}


int DTDeadFlag::setCellDead_TP( const DTWireId& id,
                                bool flag ) {
  return setCellDead_TP( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire(),
                         flag );
}


int DTDeadFlag::setCellDead_RO( int   wheelId,
                                int stationId,
                                int  sectorId,
                                int      slId,
                                int   layerId,
                                int    cellId,
                                bool flag ) {

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                             dead_HV, dead_TP, dead_RO, discCat );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                   dead_HV, dead_TP, flag, discCat );
  return status;

}


int DTDeadFlag::setCellDead_RO( const DTWireId& id,
                                bool flag ) {
  return setCellDead_RO( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire(),
                         flag );
}


int DTDeadFlag::setCellDiscCat( int   wheelId,
                                int stationId,
                                int  sectorId,
                                int      slId,
                                int   layerId,
                                int    cellId,
                                bool flag ) {

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                             dead_HV, dead_TP, dead_RO, discCat );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                   dead_HV, dead_TP, dead_RO, flag );
  return status;

}


int DTDeadFlag::setCellDiscCat( const DTWireId& id,
                                bool flag ) {
  return setCellDiscCat( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire(),
                         flag );
}


DTDeadFlag::const_iterator DTDeadFlag::begin() const {
  return cellData.begin();
}


DTDeadFlag::const_iterator DTDeadFlag::end() const {
  return cellData.end();
}


