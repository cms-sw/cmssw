/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/07/19 09:32:10 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTPerformance.h"

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
DTPerformance::DTPerformance():
  dataVersion( " " ) {
}


DTPerformance::DTPerformance( const std::string& version ):
  dataVersion( version ) {
}


DTPerformanceId::DTPerformanceId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ) {
}


DTPerformanceData::DTPerformanceData() :
  meanT0(         0.0 ),
  meanTtrig(      0.0 ),
  meanMtime(      0.0 ),
  meanNoise(      0.0 ),
  meanAfterPulse( 0.0 ),
  meanResolution( 0.0 ),
  meanEfficiency( 0.0 ) {
}


//--------------
// Destructor --
//--------------
DTPerformance::~DTPerformance() {
}


DTPerformanceData::~DTPerformanceData() {
}


DTPerformanceId::~DTPerformanceId() {
}


//--------------
// Operations --
//--------------
bool DTPerformanceCompare::operator()( const DTPerformanceId& idl,
                                       const DTPerformanceId& idr ) const {
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


int DTPerformance::slPerformance( int   wheelId,
                                  int stationId,
                                  int  sectorId,
                                  int      slId,
                                  float& meanT0,
                                  float& meanTtrig,
                                  float& meanMtime,
                                  float& meanNoise,
                                  float& meanAfterPulse,
                                  float& meanResolution,
                                  float& meanEfficiency,
                                  DTTimeUnits::type unit ) const {

  meanT0         = 0.0;
  meanTtrig      = 0.0;
  meanMtime      = 0.0;
  meanNoise      = 0.0;
  meanAfterPulse = 0.0;
  meanResolution = 0.0;
  meanEfficiency = 0.0;

  DTPerformanceId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;
  std::map<DTPerformanceId,
           DTPerformanceData,
           DTPerformanceCompare>::const_iterator iter = slData.find( key );

  if ( iter != slData.end() ) {
    const DTPerformanceData& data = iter->second;
    meanT0         = data.meanT0;
    meanTtrig      = data.meanTtrig;
    meanMtime      = data.meanMtime;
    meanNoise      = data.meanNoise;
    meanAfterPulse = data.meanAfterPulse;
    meanResolution = data.meanResolution;
    meanEfficiency = data.meanEfficiency;
    if ( unit == DTTimeUnits::ns ) {
      meanT0    *= nsPerCount;
      meanTtrig *= nsPerCount;
      meanMtime *= nsPerCount;
    }
    return 0;
  }
  return 1;

}


int DTPerformance::slPerformance( const DTSuperLayerId& id,
                                  float& meanT0,
                                  float& meanTtrig,
                                  float& meanMtime,
                                  float& meanNoise,
                                  float& meanAfterPulse,
                                  float& meanResolution,
                                  float& meanEfficiency,
                                  DTTimeUnits::type unit ) const {
  return slPerformance( id.wheel(),
                        id.station(),
                        id.sector(),
                        id.superLayer(),
                        meanT0,
                        meanTtrig,
                        meanMtime,
                        meanNoise,
                        meanAfterPulse,
                        meanResolution,
                        meanEfficiency,
                        unit );
}


float DTPerformance::unit() const {
  return nsPerCount;
}


const
std::string& DTPerformance::version() const {
  return dataVersion;
}


std::string& DTPerformance::version() {
  return dataVersion;
}


void DTPerformance::clear() {
  slData.clear();
  return;
}


int DTPerformance::setSLPerformance( int   wheelId,
                                     int stationId,
                                     int  sectorId,
                                     int      slId,
                                     float meanT0,
                                     float meanTtrig,
                                     float meanMtime,
                                     float meanNoise,
                                     float meanAfterPulse,
                                     float meanResolution,
                                     float meanEfficiency,
                                     DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    meanT0    /= nsPerCount;
    meanTtrig /= nsPerCount;
    meanMtime /= nsPerCount;
  }

  DTPerformanceId key;
  key.  wheelId =   wheelId;
  key.stationId = stationId;
  key. sectorId =  sectorId;
  key.     slId =      slId;

  std::map<DTPerformanceId,
           DTPerformanceData,
           DTPerformanceCompare>::iterator iter = slData.find( key );
  if ( iter != slData.end() ) {
    DTPerformanceData& data = iter->second;
    data.meanNoise      = meanNoise;
    data.meanAfterPulse = meanAfterPulse;
    data.meanResolution = meanResolution;
    data.meanEfficiency = meanEfficiency;
  }
  else {
    DTPerformanceData data;
    data.meanNoise      = meanNoise;
    data.meanAfterPulse = meanAfterPulse;
    data.meanResolution = meanResolution;
    data.meanEfficiency = meanEfficiency;
    slData.insert( std::pair<const DTPerformanceId,DTPerformanceData>( key, data ) );
  }

  return 0;

}


int DTPerformance::setSLPerformance( const DTSuperLayerId& id,
                                     float meanT0,
                                     float meanTtrig,
                                     float meanMtime,
                                     float meanNoise,
                                     float meanAfterPulse,
                                     float meanResolution,
                                     float meanEfficiency,
                                     DTTimeUnits::type unit ) {
  return setSLPerformance( id.wheel(),
                           id.station(),
                           id.sector(),
                           id.superLayer(),
                           meanT0,
                           meanTtrig,
                           meanMtime,
                           meanNoise,
                           meanAfterPulse,
                           meanResolution,
                           meanEfficiency,
                           unit );
}


void DTPerformance::setUnit( float unit ) {
  nsPerCount = unit;
}


DTPerformance::const_iterator DTPerformance::begin() const {
  return slData.begin();
}


DTPerformance::const_iterator DTPerformance::end() const {
  return slData.end();
}


