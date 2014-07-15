/*
 *  See header file for a description of this class.
 *
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
#include "CondFormats/DTObjects/interface/DTBufferTree.h"

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
DTPerformance::DTPerformance():
  dataVersion( " " ),
  nsPerCount(0.0),
  dBuf(new DTBufferTree<int,int>) {
  dataList.reserve( 1000 );
}


DTPerformance::DTPerformance( const std::string& version ):
  dataVersion( version ),
  nsPerCount(0.0),
  dBuf(new DTBufferTree<int,int>)  {
  dataList.reserve( 1000 );
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
int DTPerformance::get( int   wheelId,
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

  meanT0         =
  meanTtrig      =
  meanMtime      =
  meanNoise      =
  meanAfterPulse =
  meanResolution =
  meanEfficiency = 0.0;

  std::vector<int> chanKey;
  chanKey.reserve(4);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  chanKey.push_back(      slId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );
  if ( !searchStatus ) {
    const DTPerformanceData& data( dataList[ientry].second );
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
  }

  return searchStatus;

}


int DTPerformance::get( const DTSuperLayerId& id,
                        float& meanT0,
                        float& meanTtrig,
                        float& meanMtime,
                        float& meanNoise,
                        float& meanAfterPulse,
                        float& meanResolution,
                        float& meanEfficiency,
                        DTTimeUnits::type unit ) const {
  return get( id.wheel(),
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
  dataList.clear();
  initialize();
  return;
}


int DTPerformance::set( int   wheelId,
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

  std::vector<int> chanKey;
  chanKey.reserve(4);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  chanKey.push_back(      slId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );

  if ( !searchStatus ) {
    DTPerformanceData& data( dataList[ientry].second );
    data.meanT0         = meanT0;
    data.meanTtrig      = meanTtrig;
    data.meanMtime      = meanMtime;
    data.meanNoise      = meanNoise;
    data.meanAfterPulse = meanAfterPulse;
    data.meanResolution = meanResolution;
    data.meanEfficiency = meanEfficiency;
    return -1;
  }
  else {
    DTPerformanceId key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    key.     slId =      slId;
    DTPerformanceData data;
    data.meanT0         = meanT0;
    data.meanTtrig      = meanTtrig;
    data.meanMtime      = meanMtime;
    data.meanNoise      = meanNoise;
    data.meanAfterPulse = meanAfterPulse;
    data.meanResolution = meanResolution;
    data.meanEfficiency = meanEfficiency;
    ientry = dataList.size();
    dataList.push_back( std::pair<DTPerformanceId,DTPerformanceData>(
                        key, data ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
    return 0;
  }

  return 99;

}


int DTPerformance::set( const DTSuperLayerId& id,
                        float meanT0,
                        float meanTtrig,
                        float meanMtime,
                        float meanNoise,
                        float meanAfterPulse,
                        float meanResolution,
                        float meanEfficiency,
                        DTTimeUnits::type unit ) {
  return set( id.wheel(),
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
  return dataList.begin();
}


DTPerformance::const_iterator DTPerformance::end() const {
  return dataList.end();
}


std::string DTPerformance::mapName() const {
  std::stringstream name;
  name << dataVersion << "_map_Performance" << this;
  return name.str();
}


void DTPerformance::initialize() {

  dBuf->clear();

  int entryNum = 0;
  int entryMax = dataList.size();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  while ( entryNum < entryMax ) {

    const DTPerformanceId& chan = dataList[entryNum].first;

    chanKey.clear();
    chanKey.push_back( chan.  wheelId );
    chanKey.push_back( chan.stationId );
    chanKey.push_back( chan. sectorId );
    chanKey.push_back( chan.     slId );
    dBuf->insert( chanKey.begin(), chanKey.end(), entryNum++ );
  }
  return;
}
