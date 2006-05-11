/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/02 16:00:00 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTRangeT0.h"
#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

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

DTSLRangeT0Data::DTSLRangeT0Data() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
      t0min( 0 ),
      t0max( 0 ) {
}

//--------------
// Destructor --
//--------------
DTRangeT0::~DTRangeT0() {
  std::string rangeT0VersionMin = dataVersion + "_rangeT0Min";
  std::string rangeT0VersionMax = dataVersion + "_rangeT0Max";
  DTDataBuffer<int,int>::dropBuffer( rangeT0VersionMin );
  DTDataBuffer<int,int>::dropBuffer( rangeT0VersionMax );
}

DTSLRangeT0Data::~DTSLRangeT0Data() {
}

//--------------
// Operations --
//--------------
int DTRangeT0::slRangeT0( int   wheelId,
                          int stationId,
                          int  sectorId,
                          int      slId,
                          int&    t0min,
                          int&    t0max ) const {

  t0min = 0;
  t0max = 0;

  std::string rangeT0VersionMin = dataVersion + "_rangeT0Min";
  std::string rangeT0VersionMax = dataVersion + "_rangeT0Max";
  DTBufferTree<int,int>* minT0Buf =
  DTDataBuffer<int,int>::findBuffer( rangeT0VersionMin );
  DTBufferTree<int,int>* maxT0Buf =
  DTDataBuffer<int,int>::findBuffer( rangeT0VersionMax );

  if ( minT0Buf == 0 ) {
    initSetup();
    minT0Buf = DTDataBuffer<int,int>::findBuffer( rangeT0VersionMin );
  }

  if ( maxT0Buf == 0 ) {
    initSetup();
    maxT0Buf = DTDataBuffer<int,int>::findBuffer( rangeT0VersionMax );
  }

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );
  t0min = minT0Buf->find( slKey.begin(), slKey.end() );
  t0max = maxT0Buf->find( slKey.begin(), slKey.end() );

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

  DTSLRangeT0Data data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.    t0min =     t0min;
  data.    t0max =     t0max;

  slData.push_back( data );

  std::string rangeT0VersionMin = dataVersion + "_rangeT0Min";
  std::string rangeT0VersionMax = dataVersion + "_rangeT0Max";

  DTBufferTree<int,int>* minT0Buf =
  DTDataBuffer<int,int>::findBuffer( rangeT0VersionMin );
  DTBufferTree<int,int>* maxT0Buf =
  DTDataBuffer<int,int>::findBuffer( rangeT0VersionMax );

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );

  minT0Buf->insert( slKey.begin(), slKey.end(), t0min );
  maxT0Buf->insert( slKey.begin(), slKey.end(), t0max );

  return 0;

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


void DTRangeT0::initSetup() const {

  std::string rangeT0VersionMin = dataVersion + "_rangeT0Min";
  std::string rangeT0VersionMax = dataVersion + "_rangeT0Max";

  DTBufferTree<int,int>* minT0Buf =
  DTDataBuffer<int,int>::findBuffer( rangeT0VersionMin );
  DTBufferTree<int,int>* maxT0Buf =
  DTDataBuffer<int,int>::findBuffer( rangeT0VersionMax );

  std::vector<DTSLRangeT0Data>::const_iterator iter = slData.begin();
  std::vector<DTSLRangeT0Data>::const_iterator iend = slData.end();
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int     t0min;
  int     t0max;
  while ( iter != iend ) {

    const DTSLRangeT0Data& data = *iter++;
      wheelId = data.  wheelId;
    stationId = data.stationId;
     sectorId = data. sectorId;
         slId = data.     slId;

    std::vector<int> slKey;
    slKey.push_back(   wheelId );
    slKey.push_back( stationId );
    slKey.push_back(  sectorId );
    slKey.push_back(      slId );

    t0min = data.t0min;
    t0max = data.t0max;
    minT0Buf->insert( slKey.begin(), slKey.end(), t0min );
    maxT0Buf->insert( slKey.begin(), slKey.end(), t0max );

  }

  return;

}

