/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/02/24 18:28:06 $
 *  $Revision: 1.3 $
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
int DTMtime::rmsFactor = 1000;

//----------------
// Constructors --
//----------------
DTMtime::DTMtime():
 dataVersion( " " ) {
}

DTMtime::DTMtime( const std::string& version ):
 dataVersion( version ) {
}

DTSLMtimeData::DTSLMtimeData() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
      mTime( 0 ),
      mTrms( 0 ) {
}

//--------------
// Destructor --
//--------------
DTMtime::~DTMtime() {
}

DTSLMtimeData::~DTSLMtimeData() {
}

//--------------
// Operations --
//--------------
void DTMtime::initSetup() const {

  std::string mTimeVersion = dataVersion + "_mTime";

  DTBufferTree<int,int  >* dataBuf =
                           DTDataBuffer<int,int  >::openBuffer( mTimeVersion );
  DTBufferTree<int,float>* drmsBuf =
                           DTDataBuffer<int,float>::openBuffer( mTimeVersion );

  std::vector<DTSLMtimeData>::const_iterator iter = slData.begin();
  std::vector<DTSLMtimeData>::const_iterator iend = slData.end();
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   mTime;
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
    mTrms = data.mTrms / rmsFactor;
    drmsBuf->insert( slKey.begin(), slKey.end(), mTrms );

  }

  return;

}


int DTMtime::slMtime( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int&    mTime,
                      float&  mTrms ) const {

  mTime = 0;
  mTrms = 0.0;

  std::string mTimeVersion = dataVersion + "_mTime";
  DTBufferTree<int,int  >* dataBuf =
                           DTDataBuffer<int,int  >::findBuffer( mTimeVersion );
  DTBufferTree<int,float>* drmsBuf =
                           DTDataBuffer<int,float>::findBuffer( mTimeVersion );
  if ( dataBuf == 0 ) {
    initSetup();
    dataBuf = DTDataBuffer<int,int  >::findBuffer( mTimeVersion );
  }
  if ( drmsBuf == 0 ) {
    initSetup();
    drmsBuf = DTDataBuffer<int,float>::findBuffer( mTimeVersion );
  }

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );
  mTime = dataBuf->find( slKey.begin(), slKey.end() );
  mTrms = drmsBuf->find( slKey.begin(), slKey.end() );

  return 1;

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
                         int     mTime,
                         float   mTrms ) {

  DTSLMtimeData data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.mTime = mTime;
  data.mTrms = static_cast<int>( mTrms * rmsFactor );

  slData.push_back( data );

  std::string mTimeVersion = dataVersion + "_mTime";

  DTBufferTree<int,int  >* dataBuf =
                           DTDataBuffer<int,int  >::openBuffer( mTimeVersion );
  DTBufferTree<int,float>* drmsBuf =
                           DTDataBuffer<int,float>::openBuffer( mTimeVersion );

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );

  dataBuf->insert( slKey.begin(), slKey.end(), mTime );
  drmsBuf->insert( slKey.begin(), slKey.end(), mTrms );

  return 0;

}


DTMtime::const_iterator DTMtime::begin() const {
  return slData.begin();
}


DTMtime::const_iterator DTMtime::end() const {
  return slData.end();
}

