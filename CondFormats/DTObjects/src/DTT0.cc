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
#include "CondFormats/DTObjects/interface/DTT0.h"

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
DTT0::DTT0():
  dataVersion( " " ),
  nsPerCount( 25.0 / 32.0 ) {
}


DTT0::DTT0( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ) {
}


DTT0Id::DTT0Id() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ) {
}


DTT0Data::DTT0Data() :
  t0mean( 0.0 ),
  t0rms(  0.0 ) {
}


//--------------
// Destructor --
//--------------
DTT0::~DTT0() {
  DTDataBuffer<int,int>::dropBuffer( mapName() );
}


DTT0Id::~DTT0Id() {
}


DTT0Data::~DTT0Data() {
}


//--------------
// Operations --
//--------------
int DTT0::get( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               int   layerId,
               int    cellId,
               float& t0mean,
               float& t0rms,
               DTTimeUnits::type unit ) const {

  t0mean =
  t0rms  = 0.0;

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
    const DTT0Data& data( dataList[ientry].second );
    t0mean = data.t0mean;
    t0rms  = data.t0rms;
    if ( unit == DTTimeUnits::ns ) {
      t0mean *= nsPerCount;
      t0rms  *= nsPerCount;
    }
  }

  return searchStatus;

}


int DTT0::get( const DTWireId& id,
               float& t0mean,
               float& t0rms,
               DTTimeUnits::type unit ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(),
              id.layer(),
              id.wire(),
              t0mean, t0rms, unit );
}


float DTT0::unit() const {
  return nsPerCount;
}


const
std::string& DTT0::version() const {
  return dataVersion;
}


std::string& DTT0::version() {
  return dataVersion;
}


void DTT0::clear() {
  DTDataBuffer<int,int>::dropBuffer( mapName() );
  dataList.clear();
  return;
}


int DTT0::set( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               int   layerId,
               int    cellId,
               float t0mean,
               float t0rms,
               DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    t0mean /= nsPerCount;
    t0rms  /= nsPerCount;
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
    DTT0Data& data( dataList[ientry].second );
    data.t0mean = t0mean;
    data.t0rms  = t0rms;
    return -1;
  }
  else {
    DTT0Id key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    key.     slId =      slId;
    key.  layerId =   layerId;
    key.   cellId =    cellId;
    DTT0Data data;
    data.t0mean = t0mean;
    data.t0rms  = t0rms;
    ientry = dataList.size();
    dataList.push_back( std::pair<const DTT0Id,DTT0Data>( key, data ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
    return 0;
  }

  return 99;

}


int DTT0::set( const DTWireId& id,
               float t0mean,
               float t0rms,
               DTTimeUnits::type unit ) {
  return set( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(),
              id.layer(),
              id.wire(),
              t0mean, t0rms, unit );
}


void DTT0::setUnit( float unit ) {
  nsPerCount = unit;
}


DTT0::const_iterator DTT0::begin() const {
  return dataList.begin();
}


DTT0::const_iterator DTT0::end() const {
  return dataList.end();
}


std::string DTT0::mapName() const {
/*
  std::string name = dataVersion + "_map_T0";
  char nptr[100];
  sprintf( nptr, "%x", reinterpret_cast<unsigned int>( this ) );
  name += nptr;
  return name;
*/
  std::stringstream name;
  name << dataVersion << "_map_T0" << this;
  return name.str();
}


void DTT0::cacheMap() const {

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::openBuffer( mName );

  int entryNum = 0;
  int entryMax = dataList.size();
  while ( entryNum < entryMax ) {

    const DTT0Id& chan = dataList[entryNum].first;

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

