/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/10/11 16:00:00 $
 *  $Revision: 1.1 $
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
  std::vector<DTCellT0Data>::const_iterator iter =
              cellData.begin();
  std::vector<DTCellT0Data>::const_iterator iend =
              cellData.end();
  DTDataBuffer<int>::openBuffer(   "cell", t0Version, -999 );
  DTDataBuffer<float>::openBuffer( "cell", t0Version, -999 );
  while ( iter != iend ) {
    const DTCellT0Data& data = *iter++;
    DTDataBuffer<int>::insertCellData( t0Version,
                                       data.  wheelId,
                                       data.stationId,
                                       data. sectorId,
                                       data.     slId,
                                       data.  layerId,
                                       data.   cellId,
                                       data.   t0mean,
                                       -9999999 );
    DTDataBuffer<float>::insertCellData( t0Version,
                                         data.  wheelId,
                                         data.stationId,
                                         data. sectorId,
                                         data.     slId,
                                         data.  layerId,
                                         data.   cellId,
                                         data.   t0rms / rmsFactor,
                                         -999.0 );
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

  int found =
    t0mean    = 0;
  t0rms     = 0.0;

/*
  std::vector<DTCellT0Data>::const_iterator iter = cellData.begin();
  std::vector<DTCellT0Data>::const_iterator iend = cellData.end();
  while ( iter != iend ) {
    const DTCellT0Data& data = *iter++;
    if ( data.  wheelId !=   wheelId ) continue;
    if ( data.stationId != stationId ) continue;
    if ( data. sectorId !=  sectorId ) continue;
    if ( data.     slId !=      slId ) continue;
    if ( data.  layerId !=   layerId ) continue;
    if ( data.   cellId !=    cellId ) continue;
    t0mean = data.t0mean;
    t0rms  = data.t0rms;
    t0rms /= rmsFactor;
    found = 1;
  }
*/

  std::string t0Version = dataVersion + "_t0";
  if( DTDataBuffer<int>::findBuffer( "cell", t0Version ) == 0 ) initSetup();

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

  std::vector<DTCellT0Data>::const_iterator iter = cellData.begin();
  std::vector<DTCellT0Data>::const_iterator iend = cellData.end();
  bool exist = false;
  while ( iter != iend ) {
    const DTCellT0Data& data = *iter++;
    exist = true;
    if ( ( data.  wheelId ==   wheelId ) &&
         ( data.stationId == stationId ) &&
         ( data. sectorId ==  sectorId ) &&
         ( data.     slId ==      slId ) &&
         ( data.  layerId ==   layerId ) &&
         ( data.   cellId ==    cellId ) ) break;
    exist = false;
  }
  if ( exist ) return 1;
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
  DTDataBuffer<int>::insertCellData( t0Version,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                       layerId,
                                        cellId,
                                        t0mean, -999999 );

  DTDataBuffer<float>::insertCellData( t0Version,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                       layerId,
                                        cellId,
                                         t0rms, -999.0 );

  return 0;

}


DTT0::const_iterator DTT0::begin() const {
  return cellData.begin();
}


DTT0::const_iterator DTT0::end() const {
  return cellData.end();
}

