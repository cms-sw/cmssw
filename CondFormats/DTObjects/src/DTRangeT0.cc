/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/08/22 12:46:52 $
 *  $Revision: 1.6 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTRangeT0.h"

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
DTRangeT0::DTRangeT0():
  dataVersion( " " ) {
}


DTRangeT0::DTRangeT0( const std::string& version ):
  dataVersion( version ) {
}


DTRangeT0Id::DTRangeT0Id() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ) {
}


DTRangeT0Data::DTRangeT0Data() :
  t0min( 0 ),
  t0max( 0 ) {
}


//--------------
// Destructor --
//--------------
DTRangeT0::~DTRangeT0() {
}


DTRangeT0Id::~DTRangeT0Id() {
}


DTRangeT0Data::~DTRangeT0Data() {
}


//--------------
// Operations --
//--------------
bool DTRangeT0Compare::operator()( const DTRangeT0Id& idl,
                                   const DTRangeT0Id& idr ) const {
  if ( idl.  wheelId < idr.  wheelId ) return true;
  if ( idl.  wheelId > idr.  wheelId ) return false;
  if ( idl.stationId < idr.stationId ) return true;
  if ( idl.stationId > idr.stationId ) return false;
  if ( idl. sectorId < idr. sectorId ) return true;
  if ( idl. sectorId > idr. sectorId ) return false;
  if ( idl.     slId < idr.     slId ) return true;
  if ( idl.     slId > idr.     slId ) return false;
  return false;
}


int DTRangeT0::slRangeT0( int   wheelId,
                          int stationId,
                          int  sectorId,
                          int      slId,
                          int&    t0min,
                          int&    t0max ) const {

  t0min = 0;
  t0max = 0;

  DTRangeT0Id key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  std::map<DTRangeT0Id,
           DTRangeT0Data,
           DTRangeT0Compare>::const_iterator iter = slData.find( key );

  if ( iter != slData.end() ) {
    const DTRangeT0Data& data = iter->second;
    t0min = data.t0min;
    t0max = data.t0max;
    return 0;
  }
  return 1;

}


int DTRangeT0::slRangeT0( const DTSuperLayerId& id,
                          int&    t0min,
                          int&    t0max ) const {
  return slRangeT0( id.wheel(),
                    id.station(),
                    id.sector(),
                    id.superLayer(),
                    t0min, t0max );
}


const
std::string& DTRangeT0::version() const {
  return dataVersion;
}


std::string& DTRangeT0::version() {
  return dataVersion;
}


void DTRangeT0::clear() {
  slData.clear();
  return;
}


int DTRangeT0::setSLRangeT0( int   wheelId,
                             int stationId,
                             int  sectorId,
                             int      slId,
                             int     t0min,
                             int     t0max ) {

  DTRangeT0Id key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;

  std::map<DTRangeT0Id,
           DTRangeT0Data,
           DTRangeT0Compare>::iterator iter = slData.find( key );
  if ( iter != slData.end() ) {
    DTRangeT0Data& data = iter->second;
    data.t0min = t0min;
    data.t0max = t0max;
    return -1;
  }
  else {
    DTRangeT0Data data;
    data.t0min = t0min;
    data.t0max = t0max;
    slData.insert( std::pair<const DTRangeT0Id,DTRangeT0Data>( key, data ) );
    return 0;
  }

  return 99;

}


int DTRangeT0::setSLRangeT0( const DTSuperLayerId& id,
                             int     t0min,
                             int     t0max ) {
  return setSLRangeT0( id.wheel(),
                       id.station(),
                       id.sector(),
                       id.superLayer(),
                       t0min, t0max );
}


DTRangeT0::const_iterator DTRangeT0::begin() const {
  return slData.begin();
}


DTRangeT0::const_iterator DTRangeT0::end() const {
  return slData.end();
}


