/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/08/15 12:00:00 $
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
   deadFlag( false ),
   nohvFlag( false ) {
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
                            bool& deadFlag,
                            bool& nohvFlag ) const {

  deadFlag = false;
  nohvFlag = false;

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
    deadFlag = data. deadFlag;
    nohvFlag = data. nohvFlag;
    return 0;
  }
  return 1;

}


int DTDeadFlag::cellStatus( const DTWireId& id,
                            bool&  deadFlag,
                            bool&  nohvFlag ) const {
  return cellStatus( id.wheel(),
                     id.station(),
                     id.sector(),
                     id.superLayer(),
                     id.layer(),
                     id.wire(),
                     deadFlag, nohvFlag );
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
                               bool deadFlag,
                               bool nohvFlag ) {

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
    data.deadFlag = deadFlag;
    data.nohvFlag = nohvFlag;
    return -1;
  }
  else {
    DTDeadFlagData data;
    data.deadFlag = deadFlag;
    data.nohvFlag = nohvFlag;
    cellData.insert( std::pair<const DTDeadFlagId,
                                     DTDeadFlagData>( key, data ) );
    return 0;
  }

  return 99;

}


int DTDeadFlag::setCellStatus( const DTWireId& id,
                               bool deadFlag,
                               bool nohvFlag  ) {
  return setCellStatus( id.wheel(),
                        id.station(),
                        id.sector(),
                        id.superLayer(),
                        id.layer(),
                        id.wire(),
                        deadFlag, nohvFlag );
}


int DTDeadFlag::setCellDead( int   wheelId,
                             int stationId,
                             int  sectorId,
                             int      slId,
                             int   layerId,
                             int    cellId,
                             bool flag ) {

  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                      flag,
                  nohvFlag );
  return status;

}


int DTDeadFlag::setCellDead( const DTWireId& id,
                             bool flag ) {
  return setCellDead( id.wheel(),
                      id.station(),
                      id.sector(),
                      id.superLayer(),
                      id.layer(),
                      id.wire(),
                      flag );
}


int DTDeadFlag::setCellNoHV( int   wheelId,
                             int stationId,
                             int  sectorId,
                             int      slId,
                             int   layerId,
                             int    cellId,
                             bool flag ) {

  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                  deadFlag,
                      flag );
  return status;

}


int DTDeadFlag::setCellNoHV( const DTWireId& id,
                               bool flag ) {
  return setCellNoHV( id.wheel(),
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


