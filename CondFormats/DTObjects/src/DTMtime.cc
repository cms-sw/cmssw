/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/12/01 12:42:36 $
 *  $Revision: 1.1 $
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
  int minWheel;
  int minStation;
  int minSector;
  int minSL;
  int maxWheel;
  int maxStation;
  int maxSector;
  int maxSL;
  getIdNumbers( minWheel, minStation, minSector, minSL,
                maxWheel, maxStation, maxSector, maxSL );
  DTDataBuffer<int>::openBuffer(   "superlayer", mTimeVersion,
                minWheel, minStation, minSector, minSL, 0, 0,
                maxWheel, maxStation, maxSector, maxSL, 1, 1,
                -999 );
  DTDataBuffer<float>::openBuffer( "superlayer", mTimeVersion,
                minWheel, minStation, minSector, minSL, 0, 0,
                maxWheel, maxStation, maxSector, maxSL, 1, 1,
                -999 );
  std::vector<DTSLMtimeData>::const_iterator iter = slData.begin();
  std::vector<DTSLMtimeData>::const_iterator iend = slData.end();
  while ( iter != iend ) {
    const DTSLMtimeData& data = *iter++;
    DTDataBuffer<int>::insertSLData( mTimeVersion,
                                     data.  wheelId,
                                     data.stationId,
                                     data. sectorId,
                                     data.     slId,
                                     data.    mTime );
    DTDataBuffer<float>::insertSLData( mTimeVersion,
                                     data.  wheelId,
                                     data.stationId,
                                     data. sectorId,
                                     data.     slId,
                                     data.    mTrms / rmsFactor );
  }

  return;

}


int DTMtime::slMtime( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int&    mTime,
                      float&  mTrms ) const {

  int found = 0;
  mTime = 0;
  mTrms = 0.0;

  std::string mTimeVersion = dataVersion + "_mTime";
  if( !DTDataBuffer<int>::findBuffer( "superlayer", mTimeVersion ) )
      initSetup();

  mTime = DTDataBuffer<int>::getSLData( mTimeVersion,
                                             wheelId,
                                           stationId,
                                            sectorId,
                                                slId );
  mTrms = DTDataBuffer<float>::getSLData( mTimeVersion,
                                               wheelId,
                                             stationId,
                                              sectorId,
                                                  slId );

  if ( mTrms >= 0.0 ) found = 1;
  return found;

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
  if( !DTDataBuffer<int>::findBuffer( "superlayer", mTimeVersion ) )
      return 0;
  DTDataBuffer<int>::insertSLData( mTimeVersion,
                                        wheelId,
                                      stationId,
                                       sectorId,
                                           slId,
                                          mTime );
  DTDataBuffer<float>::insertSLData( mTimeVersion,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                         mTrms );

  return 0;

}


DTMtime::const_iterator DTMtime::begin() const {
  return slData.begin();
}


DTMtime::const_iterator DTMtime::end() const {
  return slData.end();
}


void DTMtime::getIdNumbers( int& minWheel,  int& minStation,
                            int& minSector, int& minSL,
                            int& maxWheel,  int& maxStation,
                            int& maxSector, int& maxSL      ) const {

  std::vector<DTSLMtimeData>::const_iterator iter = slData.begin();
  std::vector<DTSLMtimeData>::const_iterator iend = slData.end();
  minWheel   = 99999999;
  minStation = 99999999;
  minSector  = 99999999;
  minSL      = 99999999;
  maxWheel   = 0;
  maxStation = 0;
  maxSector  = 0;
  maxSL      = 0;
  int id;
  while ( iter != iend ) {
    const DTSLMtimeData& data = *iter++;
    if ( ( id = data.  wheelId ) < minWheel   ) minWheel   = id;
    if ( ( id = data.stationId ) < minStation ) minStation = id;
    if ( ( id = data. sectorId ) < minSector  ) minSector  = id;
    if ( ( id = data.     slId ) < minSL      ) minSL      = id;
    if ( ( id = data.  wheelId ) > maxWheel   ) maxWheel   = id;
    if ( ( id = data.stationId ) > maxStation ) maxStation = id;
    if ( ( id = data. sectorId ) > maxSector  ) maxSector  = id;
    if ( ( id = data.     slId ) > maxSL      ) maxSL      = id;
  }

  return;

}

