/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/01/27 15:22:15 $
 *  $Revision: 1.2 $
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
  DTDataBuffer<int>::openBuffer(   "superlayer", tTrigVersion,
                minWheel, minStation, minSector, minSL, 0, 0,
                maxWheel, maxStation, maxSector, maxSL, 1, 1,
                -999 );
  std::vector<DTSLTtrigData>::const_iterator iter = slData.begin();
  std::vector<DTSLTtrigData>::const_iterator iend = slData.end();
  while ( iter != iend ) {
    const DTSLTtrigData& data = *iter++;
    DTDataBuffer<int>::insertSLData( tTrigVersion,
                                     data.  wheelId,
                                     data.stationId,
                                     data. sectorId,
                                     data.     slId,
                                     data.    tTrig );
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
  if( !DTDataBuffer<int>::findBuffer( "superlayer", tTrigVersion ) )
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

  DTSLTtrigData data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.    tTrig =     tTrig;

  slData.push_back( data );

  std::string tTrigVersion = dataVersion + "_tTrig";
  if( !DTDataBuffer<int>::findBuffer( "superlayer", tTrigVersion ) )
      return 0;
  return 0;
  DTDataBuffer<int>::insertSLData( tTrigVersion,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                         tTrig );

  return 0;

}


DTTtrig::const_iterator DTTtrig::begin() const {
  return slData.begin();
}


DTTtrig::const_iterator DTTtrig::end() const {
  return slData.end();
}


void DTTtrig::getIdNumbers( int& minWheel,  int& minStation,
                            int& minSector, int& minSL,
                            int& maxWheel,  int& maxStation,
                            int& maxSector, int& maxSL      ) const {

  std::vector<DTSLTtrigData>::const_iterator iter = slData.begin();
  std::vector<DTSLTtrigData>::const_iterator iend = slData.end();
  minWheel   = 99999999;
  minStation = 99999999;
  minSector  = 99999999;
  minSL      = 99999999;
  maxWheel   = 0;
  maxStation = 0;
  maxSector  = 0;
  maxSL      = 0;
  int id;
  int nfound = 0;
  while ( iter != iend ) {
    const DTSLTtrigData& data = *iter++;
    if ( ( id = data.  wheelId ) < minWheel   ) minWheel   = id;
    if ( ( id = data.stationId ) < minStation ) minStation = id;
    if ( ( id = data. sectorId ) < minSector  ) minSector  = id;
    if ( ( id = data.     slId ) < minSL      ) minSL      = id;
    if ( ( id = data.  wheelId ) > maxWheel   ) maxWheel   = id;
    if ( ( id = data.stationId ) > maxStation ) maxStation = id;
    if ( ( id = data. sectorId ) > maxSector  ) maxSector  = id;
    if ( ( id = data.     slId ) > maxSL      ) maxSL      = id;
    nfound++;
  }

  if ( nfound == 0 ) {
    minWheel   = 1;
    minStation = 1;
    minSector  = 1;
    minSL      = 1;
    maxWheel   = 0;
    maxStation = 0;
    maxSector  = 0;
    maxSL      = 0;
  }

  return;

}

