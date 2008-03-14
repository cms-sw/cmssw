/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:00:51 $
 *  $Revision: 1.12 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTMtime.h"

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
DTMtime::DTMtime():
  dataVersion( " " ),
  nsPerCount( 25.0 / 32.0 ) {
}


DTMtime::DTMtime( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ) {
}


DTMtimeId::DTMtimeId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ) {
}


DTMtimeData::DTMtimeData() :
  mTime( 0.0 ),
  mTrms( 0.0 ) {
}


//--------------
// Destructor --
//--------------
DTMtime::~DTMtime() {
  DTDataBuffer<int,int>::dropBuffer( mapName() );
}


DTMtimeId::~DTMtimeId() {
}


DTMtimeData::~DTMtimeData() {
}


//--------------
// Operations --
//--------------
int DTMtime::get( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float&  mTime,
                  float&  mTrms,
                  DTTimeUnits::type unit ) const {
  return get( wheelId, stationId, sectorId,
                 slId,         0,        0,
                mTime, mTrms, unit );

}


int DTMtime::get( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float&  mTime,
                  float&  mTrms,
                  DTTimeUnits::type unit ) const {

  mTime =
  mTrms = 0.0;

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
    const DTMtimeData& data( dataList[ientry].second );
    mTime = data.mTime;
    mTrms = data.mTrms;
    if ( unit == DTTimeUnits::ns ) {
      mTime *= nsPerCount;
      mTrms *= nsPerCount;
    }
  }

  return searchStatus;

}


int DTMtime::get( const DTSuperLayerId& id,
                  float&  mTime,
                  float&  mTrms,
                  DTTimeUnits::type unit ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(), 0, 0,
              mTime, mTrms, unit );
}


int DTMtime::get( const DetId& id,
                  float&  mTime,
                  float&  mTrms,
                  DTTimeUnits::type unit ) const {
  DTWireId wireId( id.rawId() );
  return get( wireId.wheel(),
              wireId.station(),
              wireId.sector(),
              wireId.superLayer(),
              wireId.layer(),
              wireId.wire(),
              mTime, mTrms, unit );
}


float DTMtime::unit() const {
  return nsPerCount;
}


const
std::string& DTMtime::version() const {
  return dataVersion;
}


std::string& DTMtime::version() {
  return dataVersion;
}


void DTMtime::clear() {
  DTDataBuffer<int,int>::dropBuffer( mapName() );
  dataList.clear();
  return;
}


int DTMtime::set( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit ) {
  return set( wheelId, stationId, sectorId,
                 slId,         0,        0,
                mTime, mTrms, unit );
}


int DTMtime::set( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    mTime /= nsPerCount;
    mTrms /= nsPerCount;
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
    DTMtimeData& data( dataList[ientry].second );
    data.mTime = mTime;
    data.mTrms = mTrms;
    return -1;
  }
  else {
    DTMtimeId key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    key.     slId =      slId;
    key.  layerId =   layerId;
    key.   cellId =    cellId;
    DTMtimeData data;
    data.mTime = mTime;
    data.mTrms = mTrms;
    ientry = dataList.size();
    dataList.push_back( std::pair<DTMtimeId,DTMtimeData>( key, data ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
    return 0;
  }

  return 99;

}


int DTMtime::set( const DTSuperLayerId& id,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit ) {
  return set( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(), 0, 0,
              mTime, mTrms, unit );
}


int DTMtime::set( const DetId& id,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit ) {
  DTWireId wireId( id.rawId() );
  return set( wireId.wheel(),
              wireId.station(),
              wireId.sector(),
              wireId.superLayer(),
              wireId.layer(),
              wireId.wire(),
              mTime, mTrms, unit );
}


void DTMtime::setUnit( float unit ) {
  nsPerCount = unit;
}


DTMtime::const_iterator DTMtime::begin() const {
  return dataList.begin();
}


DTMtime::const_iterator DTMtime::end() const {
  return dataList.end();
}


std::string DTMtime::mapName() const {
/*
  std::string name = dataVersion + "_map_Mtime";
  char nptr[100];
  sprintf( nptr, "%x", reinterpret_cast<unsigned int>( this ) );
  name += nptr;
  return name;
*/
  std::stringstream name;
  name << dataVersion << "_map_Mtime" << this;
  return name.str();
}


void DTMtime::cacheMap() const {

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::openBuffer( mName );

  int entryNum = 0;
  int entryMax = dataList.size();
  while ( entryNum < entryMax ) {

    const DTMtimeId& chan = dataList[entryNum].first;

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

