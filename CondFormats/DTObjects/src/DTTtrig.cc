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

  DTBufferTree<int,int>* dataBuf =
                         DTDataBuffer<int,int>::openBuffer( tTrigVersion );

  std::vector<DTSLTtrigData>::const_iterator iter = slData.begin();
  std::vector<DTSLTtrigData>::const_iterator iend = slData.end();
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int tTrig;
  while ( iter != iend ) {
    const DTSLTtrigData& data = *iter++;
      wheelId = data.  wheelId;
    stationId = data.stationId;
     sectorId = data. sectorId;
         slId = data.     slId;

    std::vector<int> slKey;
    slKey.push_back(   wheelId );
    slKey.push_back( stationId );
    slKey.push_back(  sectorId );
    slKey.push_back(      slId );

    tTrig = data.tTrig;
    dataBuf->insert( slKey.begin(), slKey.end(), tTrig );

  }

  return;

}


int DTTtrig::slTtrig( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int&    tTrig ) const {

  tTrig = 0;

  std::string tTrigVersion = dataVersion + "_tTrig";
  DTBufferTree<int,int>* dataBuf =
                         DTDataBuffer<int,int>::findBuffer( tTrigVersion );

  if ( dataBuf == 0 ) {
    initSetup();
    dataBuf = DTDataBuffer<int,int>::findBuffer( tTrigVersion );
  }

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );
  tTrig = dataBuf->find( slKey.begin(), slKey.end() );

  return 1;

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

  DTBufferTree<int,int>* dataBuf =
                         DTDataBuffer<int,int>::openBuffer( tTrigVersion );

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );

  dataBuf->insert( slKey.begin(), slKey.end(), tTrig );

  return 0;

}


DTTtrig::const_iterator DTTtrig::begin() const {
  return slData.begin();
}


DTTtrig::const_iterator DTTtrig::end() const {
  return slData.end();
}

