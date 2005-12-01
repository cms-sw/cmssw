/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/11/23 17:15:00 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTTtrig.h"
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
DTTtrig::DTTtrig():
 dataVersion( " " ) {
}

DTTtrig::DTTtrig( const std::string& version ):
 dataVersion( version ) {
}

DTSLTtrigData::DTSLTtrigData() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
      tTrig( 0 ) {
}

//--------------
// Destructor --
//--------------
DTTtrig::~DTTtrig() {
}

DTSLTtrigData::~DTSLTtrigData() {
}

//--------------
// Operations --
//--------------
void DTTtrig::initSetup() const {

  std::string tTrigVersion = dataVersion + "_tTrig";
  std::vector<DTSLTtrigData>::const_iterator iter = slData.begin();
  std::vector<DTSLTtrigData>::const_iterator iend = slData.end();
  DTDataBuffer<int>::openBuffer(   "superlayer", tTrigVersion, -999 );
  DTDataBuffer<float>::openBuffer( "superlayer", tTrigVersion, -999 );
  while ( iter != iend ) {
    const DTSLTtrigData& data = *iter++;
    DTDataBuffer<int>::insertSLData( tTrigVersion,
                                     data.  wheelId,
                                     data.stationId,
                                     data. sectorId,
                                     data.     slId,
                                     data.    tTrig,
                                     -9999999 );
  }

  return;

}


int DTTtrig::slTtrig( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int&    tTrig ) const {

  int found = 0;
  tTrig = 0;

  std::string tTrigVersion = dataVersion + "_tTrig";
  if( DTDataBuffer<int>::findBuffer( "superlayer", tTrigVersion ) == 0 )
      initSetup();

  tTrig = DTDataBuffer<int>::getSLData( tTrigVersion,
                                    wheelId,
                                  stationId,
                                   sectorId,
                                       slId );

  if ( tTrig >= -999999 ) found = 1;
  return found;

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
                         int     tTrig ) {

/*
  std::vector<DTSLTtrigData>::const_iterator iter = slData.begin();
  std::vector<DTSLTtrigData>::const_iterator iend = slData.end();
  bool exist = false;
  while ( iter != iend ) {
    const DTSLTtrigData& data = *iter++;
    exist = true;
    if ( ( data.  wheelId ==   wheelId ) &&
         ( data.stationId == stationId ) &&
         ( data. sectorId ==  sectorId ) &&
         ( data.     slId ==      slId ) ) break;
    exist = false;
  }
  if ( exist ) return 1;
*/

  DTSLTtrigData data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.    tTrig =     tTrig;

  slData.push_back( data );

  std::string tTrigVersion = dataVersion + "_tTrig";
  DTDataBuffer<int>::insertSLData( tTrigVersion,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                         tTrig, -999999 );

  return 0;

}


DTTtrig::const_iterator DTTtrig::begin() const {
  return slData.begin();
}


DTTtrig::const_iterator DTTtrig::end() const {
  return slData.end();
}

