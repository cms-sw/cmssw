/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/02/24 18:28:06 $
 *  $Revision: 1.4 $
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
int DTT0::rmsFactor = 1000;

//----------------
// Constructors --
//----------------
DTT0::DTT0():
 dataVersion( " " ) {
}

DTT0::DTT0( const std::string& version ):
 dataVersion( version ) {
}

DTCellT0Data::DTCellT0Data() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ),
     t0mean( 0 ),
     t0rms(  0 ) {
}

//--------------
// Destructor --
//--------------
DTT0::~DTT0() {
}

DTCellT0Data::~DTCellT0Data() {
}

//--------------
// Operations --
//--------------
void DTT0::initSetup() const {

  std::string t0Version = dataVersion + "_t0";

  DTBufferTree<int,int  >* dataBuf =
                           DTDataBuffer<int,int  >::openBuffer( t0Version );
  DTBufferTree<int,float>* drmsBuf =
                           DTDataBuffer<int,float>::openBuffer( t0Version );

  std::vector<DTCellT0Data>::const_iterator iter = cellData.begin();
  std::vector<DTCellT0Data>::const_iterator iend = cellData.end();
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;
  int   t0mean;
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
    t0rms  = data.t0rms / rmsFactor;
    drmsBuf->insert( cellKey.begin(), cellKey.end(), t0rms );

  }

  return;

}


int DTT0::cellT0( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  int&   t0mean,
                  float& t0rms ) const {

  t0mean    = 0;
  t0rms     = 0.0;

  std::string t0Version = dataVersion + "_t0";
  DTBufferTree<int,int  >* dataBuf =
                           DTDataBuffer<int,int  >::findBuffer( t0Version );
  DTBufferTree<int,float>* drmsBuf =
                           DTDataBuffer<int,float>::findBuffer( t0Version );
  if ( dataBuf == 0 ) {
    initSetup();
    dataBuf = DTDataBuffer<int,int  >::findBuffer( t0Version );
  }
  if ( drmsBuf == 0 ) {
    initSetup();
    drmsBuf = DTDataBuffer<int,float>::findBuffer( t0Version );
  }

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  t0mean = dataBuf->find( cellKey.begin(), cellKey.end() );
  t0rms  = drmsBuf->find( cellKey.begin(), cellKey.end() );

  return 1;

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
                     int   t0mean,
                     float t0rms ) {

  DTCellT0Data data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.  layerId =   layerId;
  data.   cellId =    cellId;
  data.t0mean = t0mean;
  data.t0rms  = static_cast<int>( t0rms * rmsFactor );

  cellData.push_back( data );

  std::string t0Version = dataVersion + "_t0";

  DTBufferTree<int,int  >* dataBuf =
                           DTDataBuffer<int,int  >::openBuffer( t0Version );
  DTBufferTree<int,float>* drmsBuf =
                           DTDataBuffer<int,float>::openBuffer( t0Version );

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


DTT0::const_iterator DTT0::begin() const {
  return cellData.begin();
}


DTT0::const_iterator DTT0::end() const {
  return cellData.end();
}

