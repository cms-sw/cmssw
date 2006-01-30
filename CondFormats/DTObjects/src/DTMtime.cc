/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/11/24 16:45:00 $
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
  std::vector<DTSLMtimeData>::const_iterator iter = slData.begin();
  std::vector<DTSLMtimeData>::const_iterator iend = slData.end();
  DTDataBuffer<int>::openBuffer(   "superlayer", mTimeVersion, -999 );
  DTDataBuffer<float>::openBuffer( "superlayer", mTimeVersion, -999 );
  while ( iter != iend ) {
    const DTSLMtimeData& data = *iter++;
    DTDataBuffer<int>::insertSLData( mTimeVersion,
                                     data.  wheelId,
                                     data.stationId,
                                     data. sectorId,
                                     data.     slId,
                                     data.    mTime,
                                     -9999999 );
    DTDataBuffer<float>::insertSLData( mTimeVersion,
                                     data.  wheelId,
                                     data.stationId,
                                     data. sectorId,
                                     data.     slId,
                                     data.    mTrms / rmsFactor,
                                     -999.0 );
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
  if( DTDataBuffer<int>::findBuffer( "superlayer", mTimeVersion ) == 0 )
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

/*
  std::vector<DTSLMtimeData>::const_iterator iter = slData.begin();
  std::vector<DTSLMtimeData>::const_iterator iend = slData.end();
  bool exist = false;
  while ( iter != iend ) {
    const DTSLMtimeData& data = *iter++;
    exist = true;
    if ( ( data.  wheelId ==   wheelId ) &&
         ( data.stationId == stationId ) &&
         ( data. sectorId ==  sectorId ) &&
         ( data.     slId ==      slId ) ) break;
    exist = false;
  }
  if ( exist ) return 1;
*/

  DTSLMtimeData data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.mTime = mTime;
  data.mTrms = static_cast<int>( mTrms * rmsFactor );

  slData.push_back( data );

  std::string mTimeVersion = dataVersion + "_mTime";
  DTDataBuffer<int>::insertSLData( mTimeVersion,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                         mTime, -999999 );
  DTDataBuffer<float>::insertSLData( mTimeVersion,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                         mTrms, -999.0 );

  return 0;

}


DTMtime::const_iterator DTMtime::begin() const {
  return slData.begin();
}


DTMtime::const_iterator DTMtime::end() const {
  return slData.end();
}

