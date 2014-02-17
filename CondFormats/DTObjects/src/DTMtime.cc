/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/20 18:20:08 $
 *  $Revision: 1.16 $
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
//#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

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
  dataList.reserve( 1000 );
  dBuf = 0;
}


DTMtime::DTMtime( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ) {
  dataList.reserve( 1000 );
  dBuf = 0;
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
//  DTDataBuffer<int,int>::dropBuffer( mapName() );
  delete dBuf;
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
                  float&  mTime,
                  float&  mTrms,
                  DTVelocityUnits::type unit ) const {
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

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer<int,int>::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();

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


int DTMtime::get( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float&  mTime,
                  float&  mTrms,
                  DTVelocityUnits::type unit ) const {
  int status = get( wheelId, stationId, sectorId,
                       slId,   layerId,   cellId,
                      mTime, mTrms, DTTimeUnits::counts );
  if ( unit == DTVelocityUnits::cm_per_count ) {
    mTime  = 2.1 / mTime;
    mTrms *= mTime;
  }
  if ( unit == DTVelocityUnits::cm_per_ns ) {
    mTime  = 2.1 / mTime;
    mTrms *= mTime;
    mTime /= nsPerCount;
  }
  return status;
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


int DTMtime::get( const DTSuperLayerId& id,
                  float&  mTime,
                  float&  mTrms,
                  DTVelocityUnits::type unit ) const {
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


int DTMtime::get( const DetId& id,
                  float&  mTime,
                  float&  mTrms,
                  DTVelocityUnits::type unit ) const {
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
//  DTDataBuffer<int,int>::dropBuffer( mapName() );
  delete dBuf;
  dBuf = 0;
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
                  float   mTime,
                  float   mTrms,
                  DTVelocityUnits::type unit ) {
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

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer<int,int>::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();
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


int DTMtime::set( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float   mTime,
                  float   mTrms,
                  DTVelocityUnits::type unit ) {
  if ( unit == DTVelocityUnits::cm_per_count ) {
    mTrms /= mTime;
    mTime  = 2.1 / mTime;
  }
  if ( unit == DTVelocityUnits::cm_per_ns ) {
    mTime *= nsPerCount;
    mTrms /= mTime;
    mTime  = 2.1 / mTime;
  }
  return set( wheelId, stationId, sectorId,
                 slId,   layerId,   cellId,
                mTime, mTrms, DTTimeUnits::counts );

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


int DTMtime::set( const DTSuperLayerId& id,
                  float   mTime,
                  float   mTrms,
                  DTVelocityUnits::type unit ) {
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


int DTMtime::set( const DetId& id,
                  float   mTime,
                  float   mTrms,
                  DTVelocityUnits::type unit ) {
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
  std::stringstream name;
  name << dataVersion << "_map_Mtime" << this;
  return name.str();
}


void DTMtime::cacheMap() const {

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::openBuffer( mName );
  DTBufferTree<int,int>** pBuf;
  pBuf = const_cast<DTBufferTree<int,int>**>( &dBuf );
  *pBuf = new DTBufferTree<int,int>;

  int entryNum = 0;
  int entryMax = dataList.size();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  while ( entryNum < entryMax ) {

    const DTMtimeId& chan = dataList[entryNum].first;

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

