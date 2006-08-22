/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/07/19 09:32:10 $
 *  $Revision: 1.10 $
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


//---------------
// C++ Headers --
//---------------
#include <iostream>

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
}


DTT0Id::~DTT0Id() {
}


DTT0Data::~DTT0Data() {
}


//--------------
// Operations --
//--------------
bool DTT0Compare::operator()( const DTT0Id& idl,
                              const DTT0Id& idr ) const {
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


int DTT0::cellT0( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float& t0mean,
                  float& t0rms,
                  DTTimeUnits::type unit ) const {

  t0mean    = 0.0;
  t0rms     = 0.0;

  DTT0Id key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  key.  layerId =   layerId;
  key.   cellId =    cellId;
  std::map<DTT0Id,
           DTT0Data,
           DTT0Compare>::const_iterator iter = cellData.find( key );

  if ( iter != cellData.end() ) {
    const DTT0Data& data = iter->second;
    t0mean = data.t0mean;
    t0rms  = data.t0rms;
    if ( unit == DTTimeUnits::ns ) {
      t0mean *= nsPerCount;
      t0rms  *= nsPerCount;
    }
    return 0;
  }
  return 1;

}


int DTT0::cellT0( const DTWireId& id,
                  float& t0mean,
                  float& t0rms,
                  DTTimeUnits::type unit ) const {
  return cellT0( id.wheel(),
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
  cellData.clear();
  return;
}


int DTT0::setCellT0( int   wheelId,
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

  DTT0Id key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  key.  layerId =   layerId;
  key.   cellId =    cellId;

  std::map<DTT0Id,
           DTT0Data,
           DTT0Compare>::iterator iter = cellData.find( key );
  if ( iter != cellData.end() ) {
    DTT0Data& data = iter->second;
    data.t0mean = t0mean;
    data.t0rms  = t0rms;
  }
  else {
    DTT0Data data;
    data.t0mean = t0mean;
    data.t0rms  = t0rms;
    cellData.insert( std::pair<const DTT0Id,DTT0Data>( key, data ) );
  }

  return 0;

}


int DTT0::setCellT0( const DTWireId& id,
                     float t0mean,
                     float t0rms,
                     DTTimeUnits::type unit ) {
  return setCellT0( id.wheel(),
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
  return cellData.begin();
}


DTT0::const_iterator DTT0::end() const {
  return cellData.end();
}


