/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/08/22 12:46:52 $
 *  $Revision: 1.10 $
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
       slId( 0 ) {
}


DTMtimeData::DTMtimeData() :
  mTime( 0.0 ),
  mTrms( 0.0 ) {
}


//--------------
// Destructor --
//--------------
DTMtime::~DTMtime() {
}


DTMtimeId::~DTMtimeId() {
}


DTMtimeData::~DTMtimeData() {
}


//--------------
// Operations --
//--------------
bool DTMtimeCompare::operator()( const DTMtimeId& idl,
                                 const DTMtimeId& idr ) const {
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


int DTMtime::slMtime( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      float&  mTime,
                      float&  mTrms,
                      DTTimeUnits::type unit ) const {

  mTime = 0.0;
  mTrms = 0.0;

  DTMtimeId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  std::map<DTMtimeId,
           DTMtimeData,
           DTMtimeCompare>::const_iterator iter = slData.find( key );

  if ( iter != slData.end() ) {
    const DTMtimeData& data = iter->second;
    mTime = data.mTime;
    mTrms = data.mTrms;
    if ( unit == DTTimeUnits::ns ) {
      mTime *= nsPerCount;
      mTrms *= nsPerCount;
    }
    return 0;
  }
  return 1;

}


int DTMtime::slMtime( const DTSuperLayerId& id,
                      float&  mTime,
                      float&  mTrms,
                      DTTimeUnits::type unit ) const {
  return slMtime( id.wheel(),
                  id.station(),
                  id.sector(),
                  id.superLayer(),
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
  slData.clear();
  return;
}


int DTMtime::setSLMtime( int   wheelId,
                         int stationId,
                         int  sectorId,
                         int      slId,
                         float   mTime,
                         float   mTrms,
                         DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    mTime /= nsPerCount;
    mTrms /= nsPerCount;
  }

  DTMtimeId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;

  std::map<DTMtimeId,
           DTMtimeData,
           DTMtimeCompare>::iterator iter = slData.find( key );
  if ( iter != slData.end() ) {
    DTMtimeData& data = iter->second;
    data.mTime = mTime;
    data.mTrms = mTrms;
    return -1;
  }
  else {
    DTMtimeData data;
    data.mTime = mTime;
    data.mTrms = mTrms;
    slData.insert( std::pair<const DTMtimeId,DTMtimeData>( key, data ) );
    return 0;
  }

  return 99;

}


int DTMtime::setSLMtime( const DTSuperLayerId& id,
                         float   mTime,
                         float   mTrms,
                         DTTimeUnits::type unit ) {
  return setSLMtime( id.wheel(),
                     id.station(),
                     id.sector(),
                     id.superLayer(),
                     mTime, mTrms, unit );
}

void DTMtime::setUnit( float unit ) {
  nsPerCount = unit;
}


DTMtime::const_iterator DTMtime::begin() const {
  return slData.begin();
}


DTMtime::const_iterator DTMtime::end() const {
  return slData.end();
}


