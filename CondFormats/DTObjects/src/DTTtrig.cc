/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:00:53 $
 *  $Revision: 1.13 $
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
#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

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
  nsPerCount( 25.0 / 32.0 ) {
}


DTTtrig::DTTtrig( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ) {
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
  tTrms( 0.0 ) {
}


//--------------
// Destructor --
//--------------
DTTtrig::~DTTtrig() {
  DTDataBuffer<int,int>::dropBuffer( mapName() );
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
                  DTTimeUnits::type unit ) const {
  return get( wheelId, stationId, sectorId,
                 slId,         0,        0,
                tTrig, tTrms, unit );

}


int DTTtrig::get( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float&  tTrig,
                  float&  tTrms,
                  DTTimeUnits::type unit ) const {

  tTrig =
  tTrms = 0.0;

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::findBuffer( mName );
  if ( dBuf == 0 ) {
    cacheMap();
    dBuf =
    DTDataBuffer<int,int>::findBuffer( mName );
  }

  std::vector<int> chanKey;
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
                  DTTimeUnits::type unit ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(), 0, 0,
              tTrig, tTrms, unit );
}


int DTTtrig::get( const DetId& id,
                  float&  tTrig,
                  float&  tTrms,
                  DTTimeUnits::type unit ) const {
  DTWireId wireId( id.rawId() );
  return get( wireId.wheel(),
              wireId.station(),
              wireId.sector(),
              wireId.superLayer(),
              wireId.layer(),
              wireId.wire(),
              tTrig, tTrms, unit );
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
  DTDataBuffer<int,int>::dropBuffer( mapName() );
  dataList.clear();
  return;
}


int DTTtrig::set( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit ) {
  return set( wheelId, stationId, sectorId,
                 slId,         0,        0,
                tTrig, tTrms, unit );

}


int DTTtrig::set( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    tTrig /= nsPerCount;
    tTrms /= nsPerCount;
  }

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::findBuffer( mName );
  if ( dBuf == 0 ) {
    cacheMap();
    dBuf =
    DTDataBuffer<int,int>::findBuffer( mName );
  }
  std::vector<int> chanKey;
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
                  DTTimeUnits::type unit ) {
  return set( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(), 0, 0,
              tTrig, tTrms, unit );
}


int DTTtrig::set( const DetId& id,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit ) {
  DTWireId wireId( id.rawId() );
  return set( wireId.wheel(),
              wireId.station(),
              wireId.sector(),
              wireId.superLayer(),
              wireId.layer(),
              wireId.wire(),
              tTrig, tTrms, unit );
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
/*
  std::string name = dataVersion + "_map_Ttrig";
  char nptr[100];
  sprintf( nptr, "%x", reinterpret_cast<unsigned int>( this ) );
  name += nptr;
  return name;
*/
  std::stringstream name;
  name << dataVersion << "_map_Ttrig" << this;
  return name.str();
}


void DTTtrig::cacheMap() const {

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::openBuffer( mName );

  int entryNum = 0;
  int entryMax = dataList.size();
  while ( entryNum < entryMax ) {

    const DTTtrigId& chan = dataList[entryNum].first;

    std::vector<int> chanKey;
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

