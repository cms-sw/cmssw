/*
 *  See header file for a description of this class.
 *
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
#include "CondFormats/DTObjects/interface/DTSequentialCellNumber.h"

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
  nsPerCount( 25.0 / 32.0 ),
  dataList( DTSequentialCellNumber::max() + 10 ) {
}


DTT0::DTT0( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ),
  dataList( DTSequentialCellNumber::max() + 10 ) {
}


DTT0Data::DTT0Data() :
  channelId( 0 ),
  t0mean( 0.0 ),
  t0rms(  0.0 ) {
}


//--------------
// Destructor --
//--------------
DTT0::~DTT0() {
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

  int seqNum = DTSequentialCellNumber::id( wheelId, stationId, sectorId,
                                              slId,   layerId,   cellId );
  if ( seqNum < 0 ) return seqNum;

  const DTT0Data& data = dataList[seqNum];
  if ( data.channelId == 0 ) return -999999999;

  t0mean = data.t0mean;
  t0rms  = data.t0rms;
  if ( unit == DTTimeUnits::ns ) {
    t0mean *= nsPerCount;
    t0rms  *= nsPerCount;
  }
  return 0;

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
  int i;
  int n = dataList.size();
  for ( i = 0; i < n; ++i ) {
    DTT0Data& data = dataList[i];
    data.channelId = 0;
    data.t0mean = data.t0rms;
  }
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

  int seqNum = DTSequentialCellNumber::id( wheelId, stationId, sectorId,
                                              slId,   layerId,   cellId );
  if ( seqNum < 0 ) return seqNum;

  DTWireId id(  wheelId, stationId, sectorId,
                   slId,   layerId,   cellId );
  DTT0Data& data = dataList[seqNum];
  data.channelId = id.rawId();
  data.t0mean    = t0mean;
  data.t0rms     = t0rms;

  return 0;

}


int DTT0::set( const DTWireId& id,
               float t0mean,
               float t0rms,
               DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    t0mean /= nsPerCount;
    t0rms  /= nsPerCount;
  }

  int seqNum = DTSequentialCellNumber::id( id.wheel(),
                                           id.station(),
                                           id.sector(),
                                           id.superLayer(),
                                           id.layer(),
                                           id.wire() );
  if ( seqNum < 0 ) return seqNum;

  DTT0Data& data = dataList[seqNum];
  data.channelId = id.rawId();
  data.t0mean    = t0mean;
  data.t0rms     = t0rms;

  return 0;

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

