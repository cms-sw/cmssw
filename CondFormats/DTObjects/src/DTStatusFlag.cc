/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/07/19 09:32:10 $
 *  $Revision: 1.4 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

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
DTStatusFlag::DTStatusFlag():
  dataVersion( " " ) {
}


DTStatusFlag::DTStatusFlag( const std::string& version ):
  dataVersion( version ) {
}


DTStatusFlagId::DTStatusFlagId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ) {
}


DTStatusFlagData::DTStatusFlagData() :
  noiseFlag( false ),
     feMask( false ),
    tdcMask( false ),
   trigMask( false ),
   deadFlag( false ),
   nohvFlag( false ) {
}


//--------------
// Destructor --
//--------------
DTStatusFlag::~DTStatusFlag() {
}


DTStatusFlagId::~DTStatusFlagId() {
}


DTStatusFlagData::~DTStatusFlagData() {
}


//--------------
// Operations --
//--------------
bool DTStatusFlagCompare::operator()( const DTStatusFlagId& idl,
                                      const DTStatusFlagId& idr ) const {
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


int DTStatusFlag::cellStatus( int   wheelId,
                              int stationId,
                              int  sectorId,
                              int      slId,
                              int   layerId,
                              int    cellId,
                              bool& noiseFlag,
                              bool&    feMask,
                              bool&   tdcMask,
                              bool&  trigMask,
                              bool&  deadFlag,
                              bool&  nohvFlag ) const {

  noiseFlag = false;
     feMask = false;
    tdcMask = false;
   deadFlag = false;
   nohvFlag = false;

  DTStatusFlagId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  key.  layerId =   layerId;
  key.   cellId =    cellId;
  std::map<DTStatusFlagId,
           DTStatusFlagData,
           DTStatusFlagCompare>::const_iterator iter = cellData.find( key );

  if ( iter != cellData.end() ) {
    const DTStatusFlagData& data = iter->second;
    noiseFlag = data.noiseFlag;
       feMask = data.   feMask;
      tdcMask = data.  tdcMask;
     deadFlag = data. deadFlag;
     nohvFlag = data. nohvFlag;
    return 0;
  }
  return 1;

}


int DTStatusFlag::cellStatus( const DTWireId& id,
                              bool& noiseFlag,
                              bool&    feMask,
                              bool&   tdcMask,
                              bool&  trigMask,
                              bool&  deadFlag,
                              bool&  nohvFlag ) const {
  return cellStatus( id.wheel(),
                     id.station(),
                     id.sector(),
                     id.superLayer(),
                     id.layer(),
                     id.wire(),
                     noiseFlag,   feMask,  tdcMask,
                      trigMask, deadFlag, nohvFlag );
}


const
std::string& DTStatusFlag::version() const {
  return dataVersion;
}


std::string& DTStatusFlag::version() {
  return dataVersion;
}


void DTStatusFlag::clear() {
  cellData.clear();
  return;
}


int DTStatusFlag::setCellStatus( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId,
                                 bool noiseFlag,
                                 bool    feMask,
                                 bool   tdcMask,
                                 bool  trigMask,
                                 bool  deadFlag,
                                 bool  nohvFlag ) {

  DTStatusFlagId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  key.  layerId =   layerId;
  key.   cellId =    cellId;

  std::map<DTStatusFlagId,
           DTStatusFlagData,
           DTStatusFlagCompare>::iterator iter = cellData.find( key );
  if ( iter != cellData.end() ) {
    DTStatusFlagData& data = iter->second;
    data.noiseFlag = noiseFlag;
    data.   feMask =    feMask;
    data.  tdcMask =   tdcMask;
    data. trigMask =  trigMask;
    data. deadFlag =  deadFlag;
    data. nohvFlag =  nohvFlag;
  }
  else {
    DTStatusFlagData data;
    data.noiseFlag = noiseFlag;
    data.   feMask =    feMask;
    data.  tdcMask =   tdcMask;
    data. trigMask =  trigMask;
    data. deadFlag =  deadFlag;
    data. nohvFlag =  nohvFlag;
    cellData.insert( std::pair<const DTStatusFlagId,
                                     DTStatusFlagData>( key, data ) );
  }

  return 0;

}


int DTStatusFlag::setCellStatus( const DTWireId& id,
                                 bool noiseFlag,
                                 bool    feMask,
                                 bool   tdcMask,
                                 bool  trigMask,
                                 bool  deadFlag,
                                 bool  nohvFlag  ) {
  return setCellStatus( id.wheel(),
                        id.station(),
                        id.sector(),
                        id.superLayer(),
                        id.layer(),
                        id.wire(),
                        noiseFlag,   feMask,  tdcMask,
                         trigMask, deadFlag, nohvFlag );
}


int DTStatusFlag::setCellNoise( int   wheelId,
                                int stationId,
                                int  sectorId,
                                int      slId,
                                int   layerId,
                                int    cellId,
                                bool flag ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                      flag,
                    feMask,
                   tdcMask,
                  trigMask,
                  deadFlag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellNoise( const DTWireId& id,
                                bool flag ) {
  return setCellNoise( id.wheel(),
                       id.station(),
                       id.sector(),
                       id.superLayer(),
                       id.layer(),
                       id.wire(),
                       flag );
}


int DTStatusFlag::setCellFEMask( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId,
                                 bool mask ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                      mask,
                   tdcMask,
                  trigMask,
                  deadFlag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellFEMask( const DTWireId& id,
                                 bool mask ) {
  return setCellFEMask( id.wheel(),
                        id.station(),
                        id.sector(),
                        id.superLayer(),
                        id.layer(),
                        id.wire(),
                        mask );
}


int DTStatusFlag::setCellTDCMask( int   wheelId,
                                  int stationId,
                                  int  sectorId,
                                  int      slId,
                                  int   layerId,
                                  int    cellId,
                                  bool mask ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                    feMask,
                      mask,
                  trigMask,
                  deadFlag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellTDCMask( const DTWireId& id,
                                  bool mask ) {
  return setCellTDCMask( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire(),
                         mask );
}


int DTStatusFlag::setCellTrigMask( int   wheelId,
                                   int stationId,
                                   int  sectorId,
                                   int      slId,
                                   int   layerId,
                                   int    cellId,
                                   bool mask ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                    feMask,
                   tdcMask,
                      mask,
                  deadFlag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellTrigMask( const DTWireId& id,
                                   bool mask ) {
  return setCellTrigMask( id.wheel(),
                          id.station(),
                          id.sector(),
                          id.superLayer(),
                          id.layer(),
                          id.wire(),
                          mask );
}


int DTStatusFlag::setCellDead( int   wheelId,
                               int stationId,
                               int  sectorId,
                               int      slId,
                               int   layerId,
                               int    cellId,
                               bool flag ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                    feMask,
                   tdcMask,
                  trigMask,
                      flag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellDead( const DTWireId& id,
                               bool flag ) {
  return setCellDead( id.wheel(),
                      id.station(),
                      id.sector(),
                      id.superLayer(),
                      id.layer(),
                      id.wire(),
                      flag );
}


int DTStatusFlag::setCellNoHV( int   wheelId,
                               int stationId,
                               int  sectorId,
                               int      slId,
                               int   layerId,
                               int    cellId,
                               bool flag ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                    feMask,
                   tdcMask,
                  trigMask,
                  deadFlag,
                      flag );
  return status;

}


int DTStatusFlag::setCellNoHV( const DTWireId& id,
                               bool flag ) {
  return setCellNoHV( id.wheel(),
                      id.station(),
                      id.sector(),
                      id.superLayer(),
                      id.layer(),
                      id.wire(),
                      flag );
}


DTStatusFlag::const_iterator DTStatusFlag::begin() const {
  return cellData.begin();
}


DTStatusFlag::const_iterator DTStatusFlag::end() const {
  return cellData.end();
}


