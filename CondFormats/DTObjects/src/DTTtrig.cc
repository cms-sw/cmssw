/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/11 09:44:53 $
 *  $Revision: 1.16 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTTtrig.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "CondFormats/DTObjects/interface/DTDataBuffer.h"
#include "CondFormats/DTObjects/interface/DTBufferTree.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <sstream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTTtrig::DTTtrig():
  dataVersion( " " ),
  nsPerCount( 25.0 / 32.0 ),
  dBuf(new DTBufferTree<int,int>) {
  dataList.reserve( 1000 );
}


DTTtrig::DTTtrig( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ),
  dBuf(new DTBufferTree<int,int>) {
  dataList.reserve( 1000 );
}


DTTtrigId::DTTtrigId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ) {
}


DTTtrigData::DTTtrigData() :
  tTrig( 0.0 ),
  tTrms( 0.0 ),
  kFact( 0.0 ) {
}


//--------------
// Destructor --
//--------------
DTTtrig::~DTTtrig() {
}


DTTtrigId::~DTTtrigId() {
}


DTTtrigData::~DTTtrigData() {
}

//--------------
// Operations --
//--------------
int DTTtrig::get( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float&  tTrig,
                  float&  tTrms,
                  float&  kFact,
                  DTTimeUnits::type unit ) const {
  return get( wheelId, stationId, sectorId,
                 slId,         0,        0,
                tTrig, tTrms, kFact, unit );

}


int DTTtrig::get( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float&  tTrig,
                  float&  tTrms,
                  float&  kFact,
                  DTTimeUnits::type unit ) const {

  tTrig =
  tTrms =
  kFact = 0.0;

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
    const DTTtrigData& data( dataList[ientry].second );
    tTrig = data.tTrig;
    tTrms = data.tTrms;
    kFact = data.kFact;
    if ( unit == DTTimeUnits::ns ) {
      tTrig *= nsPerCount;
      tTrms *= nsPerCount;
    }
  }

  return searchStatus;

}


int DTTtrig::get( const DTSuperLayerId& id,
                  float&  tTrig,
                  float&  tTrms,
                  float&  kFact,
                  DTTimeUnits::type unit ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(), 0, 0,
              tTrig, tTrms, kFact, unit );
}


int DTTtrig::get( const DetId& id,
                  float&  tTrig,
                  float&  tTrms,
                  float&  kFact,
                  DTTimeUnits::type unit ) const {
  DTWireId wireId( id.rawId() );
  return get( wireId.wheel(),
              wireId.station(),
              wireId.sector(),
              wireId.superLayer(),
              wireId.layer(),
              wireId.wire(),
              tTrig, tTrms, kFact, unit );
}


int DTTtrig::get( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float&  tTrig,
                  DTTimeUnits::type unit ) const {
  return get( wheelId, stationId, sectorId, 
                 slId,         0,        0, tTrig, unit );
}


int DTTtrig::get( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float&  tTrig,
                  DTTimeUnits::type unit ) const {
  float tMean;
  float tTrms;
  float kFact;
  int status = get( wheelId, stationId, sectorId,
                       slId,   layerId,   cellId, 
                      tMean,     tTrms,    kFact, unit );
  tTrig = tMean + ( kFact * tTrms );
  return status;
}


int DTTtrig::get( const DTSuperLayerId& id,
                  float&  tTrig,
                  DTTimeUnits::type unit ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(), 0, 0,
              tTrig, unit );
}


int DTTtrig::get( const DetId& id,
                  float&  tTrig,
                  DTTimeUnits::type unit ) const {
  DTWireId wireId( id.rawId() );
  return get( wireId.wheel(),
              wireId.station(),
              wireId.sector(),
              wireId.superLayer(),
              wireId.layer(),
              wireId.wire(),
              tTrig, unit );
}


float DTTtrig::unit() const {
  return nsPerCount;
}


const
std::string& DTTtrig::version() const {
  return dataVersion;
}


std::string& DTTtrig::version() {
  return dataVersion;
}


void DTTtrig::clear() {
  dataList.clear();
  initialize();
  return;
}


int DTTtrig::set( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float   tTrig,
                  float   tTrms,
                  float   kFact,
                  DTTimeUnits::type unit ) {
  return set( wheelId, stationId, sectorId,
                 slId,         0,        0,
                tTrig, tTrms, kFact, unit );

}


int DTTtrig::set( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float   tTrig,
                  float   tTrms,
                  float   kFact,
                  DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    tTrig /= nsPerCount;
    tTrms /= nsPerCount;
  }

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
    DTTtrigData& data( dataList[ientry].second );
    data.tTrig = tTrig;
    data.tTrms = tTrms;
    data.kFact = kFact;
    return -1;
  }
  else {
    DTTtrigId key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    key.     slId =      slId;
    key.  layerId =   layerId;
    key.   cellId =    cellId;
    DTTtrigData data;
    data.tTrig = tTrig;
    data.tTrms = tTrms;
    data.kFact = kFact;
    ientry = dataList.size();
    dataList.push_back( std::pair<DTTtrigId,DTTtrigData>( key, data ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
    return 0;
  }

  return 99;

}


int DTTtrig::set( const DTSuperLayerId& id,
                  float   tTrig,
                  float   tTrms,
                  float   kFact,
                  DTTimeUnits::type unit ) {
  return set( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(), 0, 0,
              tTrig, tTrms, kFact, unit );
}


int DTTtrig::set( const DetId& id,
                  float   tTrig,
                  float   tTrms,
                  float   kFact,
                  DTTimeUnits::type unit ) {
  DTWireId wireId( id.rawId() );
  return set( wireId.wheel(),
              wireId.station(),
              wireId.sector(),
              wireId.superLayer(),
              wireId.layer(),
              wireId.wire(),
              tTrig, tTrms, kFact, unit );
}


void DTTtrig::setUnit( float unit ) {
  nsPerCount = unit;
}


DTTtrig::const_iterator DTTtrig::begin() const {
  return dataList.begin();
}


DTTtrig::const_iterator DTTtrig::end() const {
  return dataList.end();
}


std::string DTTtrig::mapName() const {
  std::stringstream name;
  name << dataVersion << "_map_Ttrig" << this;
  return name.str();
}


void DTTtrig::initialize() {

  dBuf->clear();

  int entryNum = 0;
  int entryMax = dataList.size();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  while ( entryNum < entryMax ) {

    const DTTtrigId& chan = dataList[entryNum].first;

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
