/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/19 17:09:15 $
 *  $Revision: 1.8 $
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
  dataVersion( " " ),
  nsPerCount( 25.0 / 32.0 ) {
}

DTTtrig::DTTtrig( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ) {
}

DTSLTtrigData::DTSLTtrigData() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
      tTrig( 0.0 ),
      tTrms( 0.0 ) {
}

//--------------
// Destructor --
//--------------
DTTtrig::~DTTtrig() {
  std::string tTrigVersionM = dataVersion + "_tTrigM";
  std::string tTrigVersionR = dataVersion + "_tTrigR";
  DTDataBuffer<int,float>::dropBuffer( tTrigVersionM );
  DTDataBuffer<int,float>::dropBuffer( tTrigVersionR );
}

DTSLTtrigData::~DTSLTtrigData() {
}

//--------------
// Operations --
//--------------
int DTTtrig::slTtrig( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      float&  tTrig,
                      float&  tTrms,
                      DTTimeUnits::type unit ) const {

  tTrig = 0.0;
  tTrms = 0.0;

  std::string tTrigVersionM = dataVersion + "_tTrigM";
  std::string tTrigVersionR = dataVersion + "_tTrigR";
  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::findBuffer( tTrigVersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::findBuffer( tTrigVersionR );

  if ( dataBuf == 0 ) {
    initSetup();
    dataBuf = DTDataBuffer<int,float>::findBuffer( tTrigVersionM );
  }

  if ( drmsBuf == 0 ) {
    initSetup();
    drmsBuf = DTDataBuffer<int,float>::findBuffer( tTrigVersionR );
  }

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );
//  tTrig = dataBuf->find( slKey.begin(), slKey.end() );
//  tTrms = drmsBuf->find( slKey.begin(), slKey.end() );
  int searchStatusM = dataBuf->find( slKey.begin(), slKey.end(), tTrig );
  int searchStatusR = drmsBuf->find( slKey.begin(), slKey.end(), tTrms );

  if ( unit == DTTimeUnits::ns ) {
    tTrig *= nsPerCount;
    tTrms *= nsPerCount;
  }

//  return 1;
  return ( searchStatusM || searchStatusR );

}


int DTTtrig::slTtrig( const DTSuperLayerId& id,
                      float&  tTrig,
                      float&  tTrms,
                      DTTimeUnits::type unit ) const {
  return slTtrig( id.wheel(),
                  id.station(),
                  id.sector(),
                  id.superLayer(),
                  tTrig, tTrms, unit );
}


float DTTtrig::unit() const {
  return nsPerCount;
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
                         float   tTrig,
                         float   tTrms,
                         DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    tTrig /= nsPerCount;
    tTrms /= nsPerCount;
  }

  DTSLTtrigData data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.    tTrig =     tTrig;
  data.    tTrms =     tTrms;

  slData.push_back( data );

  std::string tTrigVersionM = dataVersion + "_tTrigM";
  std::string tTrigVersionR = dataVersion + "_tTrigR";

  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::openBuffer( tTrigVersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::openBuffer( tTrigVersionR );

  std::vector<int> slKey;
  slKey.push_back(   wheelId );
  slKey.push_back( stationId );
  slKey.push_back(  sectorId );
  slKey.push_back(      slId );

  dataBuf->insert( slKey.begin(), slKey.end(), tTrig );
  drmsBuf->insert( slKey.begin(), slKey.end(), tTrms );

  return 0;

}


int DTTtrig::setSLTtrig( const DTSuperLayerId& id,
                         float   tTrig,
                         float   tTrms,
                         DTTimeUnits::type unit ) {
  return setSLTtrig( id.wheel(),
                     id.station(),
                     id.sector(),
                     id.superLayer(),
                     tTrig, tTrms, unit );
}


void DTTtrig::setUnit( float unit ) {
  nsPerCount = unit;
}


DTTtrig::const_iterator DTTtrig::begin() const {
  return slData.begin();
}


DTTtrig::const_iterator DTTtrig::end() const {
  return slData.end();
}


void DTTtrig::initSetup() const {

  std::string tTrigVersionM = dataVersion + "_tTrigM";
  std::string tTrigVersionR = dataVersion + "_tTrigR";

  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::openBuffer( tTrigVersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::openBuffer( tTrigVersionR );

  std::vector<DTSLTtrigData>::const_iterator iter = slData.begin();
  std::vector<DTSLTtrigData>::const_iterator iend = slData.end();
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  float tTrig;
  float tTrms;
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
    tTrms = data.tTrms;
    dataBuf->insert( slKey.begin(), slKey.end(), tTrig );
    drmsBuf->insert( slKey.begin(), slKey.end(), tTrms );

  }

  return;

}

