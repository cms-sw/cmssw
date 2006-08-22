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
#include "CondFormats/DTObjects/interface/DTTtrig.h"

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
       slId( 0 ) {
}


DTTtrigData::DTTtrigData() :
  tTrig( 0.0 ),
  tTrms( 0.0 ) {
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
bool DTTtrigCompare::operator()( const DTTtrigId& idl,
                                 const DTTtrigId& idr ) const {
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


int DTTtrig::slTtrig( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      float&  tTrig,
                      float&  tTrms,
                      DTTimeUnits::type unit ) const {

  tTrig = 0.0;
  tTrms = 0.0;

  DTTtrigId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  std::map<DTTtrigId,
           DTTtrigData,
           DTTtrigCompare>::const_iterator iter = slData.find( key );

  if ( iter != slData.end() ) {
    const DTTtrigData& data = iter->second;
    tTrig = data.tTrig;
    tTrms = data.tTrms;
    if ( unit == DTTimeUnits::ns ) {
      tTrig *= nsPerCount;
      tTrms *= nsPerCount;
    }
    return 0;
  }
  return 1;

}


int DTTtrig::slTtrig( const DTSuperLayerId& id,
                      float&  tTrig,
                      float&  tTrms,
                      DTTimeUnits::type unit ) const {
  return slTtrig( id.wheel(),
                  id.station(),
                  id.sector(),
                  id.superLayer(),
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
  slData.clear();
  return;
}


int DTTtrig::setSLTtrig( int   wheelId,
                         int stationId,
                         int  sectorId,
                         int      slId,
                         float   tTrig,
                         float   tTrms,
                         DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    tTrig /= nsPerCount;
    tTrms /= nsPerCount;
  }

  DTTtrigId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;

  std::map<DTTtrigId,
           DTTtrigData,
           DTTtrigCompare>::iterator iter = slData.find( key );
  if ( iter != slData.end() ) {
    DTTtrigData& data = iter->second;
    data.tTrig = tTrig;
    data.tTrms = tTrms;
  }
  else {
    DTTtrigData data;
    data.tTrig = tTrig;
    data.tTrms = tTrms;
    slData.insert( std::pair<const DTTtrigId,DTTtrigData>( key, data ) );
  }

  return 0;

}


int DTTtrig::setSLTtrig( const DTSuperLayerId& id,
                         float   tTrig,
                         float   tTrms,
                         DTTimeUnits::type unit ) {
  return setSLTtrig( id.wheel(),
                     id.station(),
                     id.sector(),
                     id.superLayer(),
                     tTrig, tTrms, unit );
}


void DTTtrig::setUnit( float unit ) {
  nsPerCount = unit;
}


DTTtrig::const_iterator DTTtrig::begin() const {
  return slData.begin();
}


DTTtrig::const_iterator DTTtrig::end() const {
  return slData.end();
}


