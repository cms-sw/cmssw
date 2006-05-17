/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/04 06:54:26 $
 *  $Revision: 1.7 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTT0.h"
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
DTT0::DTT0():
 dataVersion( " " ),
 nsPerCount( 25.0 / 32.0 ) {
}

DTT0::DTT0( const std::string& version ):
 dataVersion( version ),
 nsPerCount( 25.0 / 32.0 ) {
}

DTCellT0Data::DTCellT0Data() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ),
     t0mean( 0.0 ),
     t0rms(  0.0 ) {
}

//--------------
// Destructor --
//--------------
DTT0::~DTT0() {
  std::string t0VersionM = dataVersion + "_t0M";
  std::string t0VersionR = dataVersion + "_t0R";
  DTDataBuffer<int,float>::dropBuffer( t0VersionM );
  DTDataBuffer<int,float>::dropBuffer( t0VersionR );
}

DTCellT0Data::~DTCellT0Data() {
}

//--------------
// Operations --
//--------------
int DTT0::cellT0( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float& t0mean,
                  float& t0rms,
                  DTTimeUnits::type unit ) const {

  t0mean    = 0.0;
  t0rms     = 0.0;

  std::string t0VersionM = dataVersion + "_t0M";
  std::string t0VersionR = dataVersion + "_t0R";
  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::findBuffer( t0VersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::findBuffer( t0VersionR );

  if ( dataBuf == 0 ) {
    initSetup();
    dataBuf = DTDataBuffer<int,float>::findBuffer( t0VersionM );
  }

  if ( drmsBuf == 0 ) {
    initSetup();
    drmsBuf = DTDataBuffer<int,float>::findBuffer( t0VersionR );
  }

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
//  t0mean = dataBuf->find( cellKey.begin(), cellKey.end() );
//  t0rms  = drmsBuf->find( cellKey.begin(), cellKey.end() );
  int searchStatusM = dataBuf->find( cellKey.begin(), cellKey.end(), t0mean );
  int searchStatusR = drmsBuf->find( cellKey.begin(), cellKey.end(), t0rms  );

  if ( unit == DTTimeUnits::ns ) {
    t0mean *= nsPerCount;
    t0rms  *= nsPerCount;
  }

//  return 1;
  return ( searchStatusM || searchStatusR );

}


int DTT0::cellT0( const DTWireId& id,
                  float& t0mean,
                  float& t0rms,
                  DTTimeUnits::type unit ) const {
  return cellT0( id.wheel(),
                 id.station(),
                 id.sector(),
                 id.superLayer(),
                 id.layer(),
                 id.wire(),
                 t0mean, t0rms, unit );
}


float DTT0::unit() const {
  return nsPerCount;
}


const
std::string& DTT0::version() const {
  return dataVersion;
}


std::string& DTT0::version() {
  return dataVersion;
}


void DTT0::clear() {
  cellData.clear();
  return;
}


int DTT0::setCellT0( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    cellId,
                     float t0mean,
                     float t0rms,
                     DTTimeUnits::type unit ) {

  if ( unit == DTTimeUnits::ns ) {
    t0mean /= nsPerCount;
    t0rms  /= nsPerCount;
  }

  DTCellT0Data data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.  layerId =   layerId;
  data.   cellId =    cellId;
  data.t0mean = t0mean;
  data.t0rms  = t0rms;

  cellData.push_back( data );

  std::string t0VersionM = dataVersion + "_t0M";
  std::string t0VersionR = dataVersion + "_t0R";

  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::openBuffer( t0VersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::openBuffer( t0VersionR );

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );

  dataBuf->insert( cellKey.begin(), cellKey.end(), t0mean );
  drmsBuf->insert( cellKey.begin(), cellKey.end(), t0rms  );

  return 0;

}


int DTT0::setCellT0( const DTWireId& id,
                     float t0mean,
                     float t0rms,
                     DTTimeUnits::type unit ) {
  return setCellT0( id.wheel(),
                    id.station(),
                    id.sector(),
                    id.superLayer(),
                    id.layer(),
                    id.wire(),
                    t0mean, t0rms, unit );
}


void DTT0::setUnit( float unit ) {
  nsPerCount = unit;
}


DTT0::const_iterator DTT0::begin() const {
  return cellData.begin();
}


DTT0::const_iterator DTT0::end() const {
  return cellData.end();
}


void DTT0::initSetup() const {

  std::string t0VersionM = dataVersion + "_t0M";
  std::string t0VersionR = dataVersion + "_t0R";

  DTBufferTree<int,float>* dataBuf =
  DTDataBuffer<int,float>::openBuffer( t0VersionM );
  DTBufferTree<int,float>* drmsBuf =
  DTDataBuffer<int,float>::openBuffer( t0VersionR );

  std::vector<DTCellT0Data>::const_iterator iter = cellData.begin();
  std::vector<DTCellT0Data>::const_iterator iend = cellData.end();
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;
  float t0mean;
  float t0rms;
  while ( iter != iend ) {

    const DTCellT0Data& data = *iter++;
      wheelId = data.  wheelId;
    stationId = data.stationId;
     sectorId = data. sectorId;
         slId = data.     slId;
      layerId = data.  layerId;
       cellId = data.   cellId;

    std::vector<int> cellKey;
    cellKey.push_back(   wheelId );
    cellKey.push_back( stationId );
    cellKey.push_back(  sectorId );
    cellKey.push_back(      slId );
    cellKey.push_back(   layerId );
    cellKey.push_back(    cellId );

    t0mean = data.t0mean;
    dataBuf->insert( cellKey.begin(), cellKey.end(), t0mean );
    t0rms  = data.t0rms;
    drmsBuf->insert( cellKey.begin(), cellKey.end(), t0rms );

  }

  return;

}

