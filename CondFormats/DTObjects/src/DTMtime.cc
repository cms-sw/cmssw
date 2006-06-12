/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/17 10:34:24 $
 *  $Revision: 1.7 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTMtime.h"
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
DTMtime::DTMtime():
  dataVersion( " " ),
  nsPerCount( 25.0 / 32.0 ) {
}

DTMtime::DTMtime( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ) {
}

DTSLMtimeData::DTSLMtimeData() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
      mTime( 0.0 ),
      mTrms( 0.0 ) {
}

//--------------
// Destructor --
//--------------
DTMtime::~DTMtime() {
  std::string mTimeVersionM = dataVersion + "_mTimeM";
  std::string mTimeVersionR = dataVersion + "_mTimeR";
  DTDataBuffer<int,float>::dropBuffer( mTimeVersionM );
  DTDataBuffer<int,float>::dropBuffer( mTimeVersionR );
}

DTSLMtimeData::~DTSLMtimeData() {
}

//--------------
// Operations --
//--------------
int DTMtime::slMtime( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      float&  mTime,
                      float&  mTrms,
                      DTTimeUnits::type unit ) const {

  mTime = 0.0;
  mTrms = 0.0;

  std::string mTimeVersionM = dataVersion + "_mTimeM";
  std::string mTimeVersionR = dataVersion + "_mTimeR";
  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::findBuffer( mTimeVersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::findBuffer( mTimeVersionR );
  if ( dataBuf == 0 ) {
    initSetup();
    dataBuf = DTDataBuffer<int,float>::findBuffer( mTimeVersionM );
  }
  if ( drmsBuf == 0 ) {
    initSetup();
    drmsBuf = DTDataBuffer<int,float>::findBuffer( mTimeVersionR );
  }

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );
//  mTime = dataBuf->find( slKey.begin(), slKey.end() );
//  mTrms = drmsBuf->find( slKey.begin(), slKey.end() );
  int searchStatusM = dataBuf->find( slKey.begin(), slKey.end(), mTime );
  int searchStatusR = drmsBuf->find( slKey.begin(), slKey.end(), mTrms );

  if ( unit == DTTimeUnits::ns ) {
    mTime *= nsPerCount;
    mTrms *= nsPerCount;
  }

//  return 1;
  return ( searchStatusM || searchStatusR );

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

  DTSLMtimeData data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.mTime = mTime;
  data.mTrms = mTrms;

  slData.push_back( data );

  std::string mTimeVersionM = dataVersion + "_mTimeM";
  std::string mTimeVersionR = dataVersion + "_mTimeR";

  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::openBuffer( mTimeVersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::openBuffer( mTimeVersionR );

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );

  dataBuf->insert( slKey.begin(), slKey.end(), mTime );
  drmsBuf->insert( slKey.begin(), slKey.end(), mTrms );

  return 0;

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


void DTMtime::initSetup() const {

  std::string mTimeVersionM = dataVersion + "_mTimeM";
  std::string mTimeVersionR = dataVersion + "_mTimeR";

  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::openBuffer( mTimeVersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::openBuffer( mTimeVersionR );

  std::vector<DTSLMtimeData>::const_iterator iter = slData.begin();
  std::vector<DTSLMtimeData>::const_iterator iend = slData.end();
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  float mTime;
  float mTrms;
  while ( iter != iend ) {

    const DTSLMtimeData& data = *iter++;
      wheelId = data.  wheelId;
    stationId = data.stationId;
     sectorId = data. sectorId;
         slId = data.     slId;

    std::vector<int> slKey;
    slKey.push_back(   wheelId );
    slKey.push_back( stationId );
    slKey.push_back(  sectorId );
    slKey.push_back(      slId );

    mTime = data.mTime;
    dataBuf->insert( slKey.begin(), slKey.end(), mTime );
    mTrms = data.mTrms;
    drmsBuf->insert( slKey.begin(), slKey.end(), mTrms );

  }

  return;

}

