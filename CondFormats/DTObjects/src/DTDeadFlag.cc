/*
 *  See header file for a description of this class.
 *
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
#include "CondFormats/DTObjects/interface/DTBufferTree.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <sstream>

//----------------
// Constructors --
//----------------
DTDeadFlag::DTDeadFlag():
  dataVersion( " " ),
  dBuf(new DTBufferTree<int,int>) {
  dataList.reserve( 1000 );
}


DTDeadFlag::DTDeadFlag( const std::string& version ):
  dataVersion( version ),
  dBuf(new DTBufferTree<int,int>) {
  dataList.reserve( 1000 );
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
int DTDeadFlag::get( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    cellId,
                     bool& dead_HV,
                     bool& dead_TP,
                     bool& dead_RO,
                     bool& discCat ) const {

  dead_HV =
  dead_TP =
  dead_RO =
  discCat = false;

  std::vector<int> chanKey;
  chanKey.reserve(6);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  chanKey.push_back(      slId );
  chanKey.push_back(   layerId );
  chanKey.push_back(    cellId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );
  if ( !searchStatus ) {
    const DTDeadFlagData& data( dataList[ientry].second );
    dead_HV = data.dead_HV;
    dead_TP = data.dead_TP;
    dead_RO = data.dead_RO;
    discCat = data.discCat;
  }

  return searchStatus;

}


int DTDeadFlag::get( const DTWireId& id,
                     bool& dead_HV,
                     bool& dead_TP,
                     bool& dead_RO,
                     bool& discCat ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(),
              id.layer(),
              id.wire(),
              dead_HV, dead_TP, dead_RO, discCat );
}


bool DTDeadFlag::getCellDead_HV( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId ) const {

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  get(   wheelId,
       stationId,
        sectorId,
            slId,
         layerId,
          cellId,
         dead_HV, dead_TP, dead_RO, discCat );
  return dead_HV;

}


bool DTDeadFlag::getCellDead_HV( const DTWireId& id ) const {
  return getCellDead_HV( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire() );
}


bool DTDeadFlag::getCellDead_TP( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId ) const {

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  get(   wheelId,
       stationId,
        sectorId,
            slId,
         layerId,
          cellId,
         dead_HV, dead_TP, dead_RO, discCat );
  return dead_TP;

}


bool DTDeadFlag::getCellDead_TP( const DTWireId& id ) const {
  return getCellDead_TP( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire() );
}


bool DTDeadFlag::getCellDead_RO( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId ) const {

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  get(   wheelId,
       stationId,
        sectorId,
            slId,
         layerId,
          cellId,
         dead_HV, dead_TP, dead_RO, discCat );
  return dead_RO;

}


bool DTDeadFlag::getCellDead_RO( const DTWireId& id ) const {
  return getCellDead_RO( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire() );
}


bool DTDeadFlag::getCellDiscCat( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId ) const {

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  get(   wheelId,
       stationId,
        sectorId,
            slId,
         layerId,
          cellId,
         dead_HV, dead_TP, dead_RO, discCat );
  return discCat;

}


bool DTDeadFlag::getCellDiscCat( const DTWireId& id ) const {
  return getCellDiscCat( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire() );
}


const
std::string& DTDeadFlag::version() const {
  return dataVersion;
}


std::string& DTDeadFlag::version() {
  return dataVersion;
}


void DTDeadFlag::clear() {
  dataList.clear();
  initialize();
  return;
}


int DTDeadFlag::set( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    cellId,
                     bool dead_HV,
                     bool dead_TP,
                     bool dead_RO,
                     bool discCat ) {

  std::vector<int> chanKey;
  chanKey.reserve(6);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  chanKey.push_back(      slId );
  chanKey.push_back(   layerId );
  chanKey.push_back(    cellId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );

  if ( !searchStatus ) {
    DTDeadFlagData& data( dataList[ientry].second );
    data.dead_HV = dead_HV;
    data.dead_TP = dead_TP;
    data.dead_RO = dead_RO;
    data.discCat = discCat;
    return -1;
  }
  else {
    DTDeadFlagId key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    key.     slId =      slId;
    key.  layerId =   layerId;
    key.   cellId =    cellId;
    DTDeadFlagData data;
    data.dead_HV = dead_HV;
    data.dead_TP = dead_TP;
    data.dead_RO = dead_RO;
    data.discCat = discCat;
    ientry = dataList.size();
    dataList.push_back( std::pair<const DTDeadFlagId,
                                        DTDeadFlagData>( key, data ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
    return 0;
  }

  return 99;

}


int DTDeadFlag::set( const DTWireId& id,
                     bool dead_HV,
                     bool dead_TP,
                     bool dead_RO,
                     bool discCat ) {
  return set( id.wheel(),
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
  return dataList.begin();
}


DTDeadFlag::const_iterator DTDeadFlag::end() const {
  return dataList.end();
}


std::string DTDeadFlag::mapName() const {
  std::stringstream name;
  name << dataVersion << "_map_DeadFlag" << this;
  return name.str();
}


void DTDeadFlag::initialize() {

  dBuf->clear();

  int entryNum = 0;
  int entryMax = dataList.size();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  while ( entryNum < entryMax ) {

    const DTDeadFlagId& chan = dataList[entryNum].first;

    chanKey.clear();
    chanKey.push_back( chan.  wheelId );
    chanKey.push_back( chan.stationId );
    chanKey.push_back( chan. sectorId );
    chanKey.push_back( chan.     slId );
    chanKey.push_back( chan.  layerId );
    chanKey.push_back( chan.   cellId );
    dBuf->insert( chanKey.begin(), chanKey.end(), entryNum++ );

  }

  return;

}

