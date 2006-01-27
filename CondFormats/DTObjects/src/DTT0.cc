/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/12/01 12:42:36 $
 *  $Revision: 1.2 $
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
  int minWheel;
  int minStation;
  int minSector;
  int minSL;
  int minLayer;
  int minCell;
  int maxWheel;
  int maxStation;
  int maxSector;
  int maxSL;
  int maxLayer;
  int maxCell;
  getIdNumbers( minWheel, minStation, minSector, minSL,  minLayer,   minCell,
                maxWheel, maxStation, maxSector, maxSL,  maxLayer,   maxCell );
  DTDataBuffer<int>::openBuffer(   "cell", t0Version,
                minWheel, minStation, minSector, minSL,  minLayer,   minCell,
                maxWheel, maxStation, maxSector, maxSL,  maxLayer,   maxCell,
                -999 );
  DTDataBuffer<float>::openBuffer( "cell", t0Version,
                minWheel, minStation, minSector, minSL,  minLayer,   minCell,
                maxWheel, maxStation, maxSector, maxSL,  maxLayer,   maxCell,
                -999.0 );
  std::vector<DTCellT0Data>::const_iterator iter = cellData.begin();
  std::vector<DTCellT0Data>::const_iterator iend = cellData.end();
  while ( iter != iend ) {
    const DTCellT0Data& data = *iter++;
    DTDataBuffer<int>::insertCellData( t0Version,
                                       data.  wheelId,
                                       data.stationId,
                                       data. sectorId,
                                       data.     slId,
                                       data.  layerId,
                                       data.   cellId,
                                       data.   t0mean );
    DTDataBuffer<float>::insertCellData( t0Version,
                                         data.  wheelId,
                                         data.stationId,
                                         data. sectorId,
                                         data.     slId,
                                         data.  layerId,
                                         data.   cellId,
                                         data.   t0rms / rmsFactor );
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

  int found = 0;
  t0mean    = 0;
  t0rms     = 0.0;

  std::string t0Version = dataVersion + "_t0";
  if( !DTDataBuffer<int>::findBuffer( "cell", t0Version ) ) initSetup();
  t0mean = DTDataBuffer<int>::getCellData( t0Version,
                                             wheelId,
                                           stationId,
                                            sectorId,
                                                slId,
                                             layerId,
                                              cellId );
  t0rms = DTDataBuffer<float>::getCellData( t0Version,
                                              wheelId,
                                            stationId,
                                             sectorId,
                                                 slId,
                                              layerId,
                                               cellId );

  if ( t0rms >= 0.0 ) found = 1;
  return found;

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
  if( !DTDataBuffer<int>::findBuffer( "cell", t0Version ) ) return 0;
  DTDataBuffer<int>::insertCellData( t0Version,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                       layerId,
                                        cellId,
                                        t0mean );
  DTDataBuffer<float>::insertCellData( t0Version,
                                         wheelId,
                                       stationId,
                                        sectorId,
                                            slId,
                                         layerId,
                                          cellId,
                                           t0rms );

  return 0;

}


DTT0::const_iterator DTT0::begin() const {
  return cellData.begin();
}


DTT0::const_iterator DTT0::end() const {
  return cellData.end();
}


void DTT0::getIdNumbers( int& minWheel,   int& minStation,
                         int& minSector,  int& minSL,
                         int& minLayer,   int& minCell,
                         int& maxWheel,   int& maxStation,
                         int& maxSector,  int& maxSL,
                         int& maxLayer,   int& maxCell    ) const {

  std::vector<DTCellT0Data>::const_iterator iter = cellData.begin();
  std::vector<DTCellT0Data>::const_iterator iend = cellData.end();
  minWheel   = 99999999;
  minStation = 99999999;
  minSector  = 99999999;
  minSL      = 99999999;
  minLayer   = 99999999;
  minCell    = 99999999;
  maxWheel   = 0;
  maxStation = 0;
  maxSector  = 0;
  maxSL      = 0;
  maxLayer   = 0;
  maxCell    = 0;
  int id;
  while ( iter != iend ) {
    const DTCellT0Data& data = *iter++;
    if ( ( id = data.  wheelId ) < minWheel   ) minWheel   = id;
    if ( ( id = data.stationId ) < minStation ) minStation = id;
    if ( ( id = data. sectorId ) < minSector  ) minSector  = id;
    if ( ( id = data.     slId ) < minSL      ) minSL      = id;
    if ( ( id = data.  layerId ) < minLayer   ) minLayer   = id;
    if ( ( id = data.   cellId ) < minCell    ) minCell    = id;
    if ( ( id = data.  wheelId ) > maxWheel   ) maxWheel   = id;
    if ( ( id = data.stationId ) > maxStation ) maxStation = id;
    if ( ( id = data. sectorId ) > maxSector  ) maxSector  = id;
    if ( ( id = data.     slId ) > maxSL      ) maxSL      = id;
    if ( ( id = data.  layerId ) > maxLayer   ) maxLayer   = id;
    if ( ( id = data.   cellId ) > maxCell    ) maxCell    = id;
  }

  return;

}

