/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/20 18:20:08 $
 *  $Revision: 1.17 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTT0.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <sstream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTT0::DTT0():
  dataVersion( " " ),
  nsPerCount( 25.0 / 32.0 ) {
  dataList.reserve( 12000 );
  dBuf = sortedLayers = 0;
}


DTT0::DTT0( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ) {
  dataList.reserve( 12000 );
  dBuf = sortedLayers = 0;
}


DTT0Id::DTT0Id() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ) {
}


DTT0Data::DTT0Data() :
  t0mean( 0.0 ),
  t0rms(  0.0 ) {
}


//--------------
// Destructor --
//--------------
DTT0::~DTT0() {
  delete dBuf;
  delete sortedLayers;
}


DTT0Id::~DTT0Id() {
}


DTT0Data::~DTT0Data() {
}


//--------------
// Operations --
//--------------
int DTT0::get( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               int   layerId,
               int    cellId,
               float& t0mean,
               float& t0rms,
               DTTimeUnits::type unit ) const {

  t0mean =
  t0rms  = 0.0;

  if ( ( dBuf == 0 ) && ( sortedLayers == 0 ) ) checkOrder();
  if ( sortedLayers != 0 ) return getSorted( wheelId, stationId, sectorId,
                                                slId,   layerId,   cellId,
                                              t0mean,     t0rms,     unit );
  else                     return getRandom( wheelId, stationId, sectorId,
                                                slId,   layerId,   cellId,
                                              t0mean,     t0rms,     unit );
  return -999999999;
}


int DTT0::getRandom( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    cellId,
                     float& t0mean,
                     float& t0rms,
                     DTTimeUnits::type unit ) const {

  DTWireId detId( wheelId, stationId, sectorId,
                  slId,   layerId,   cellId );
  int chanKey = detId.rawId();

  int ientry;
  std::map<int,int>::const_iterator buf_iter = dBuf->find( chanKey );
  std::map<int,int>::const_iterator buf_iend = dBuf->end();
  int searchStatus = ( buf_iter == buf_iend );
  if ( !searchStatus ) {
    ientry = buf_iter->second;
    const DTT0Data& data( dataList[ientry].second );
    t0mean = data.t0mean;
    t0rms  = data.t0rms;
    if ( unit == DTTimeUnits::ns ) {
      t0mean *= nsPerCount;
      t0rms  *= nsPerCount;
    }
  }

  return searchStatus;

}


int DTT0::getSorted( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    cellId,
                     float& t0mean,
                     float& t0rms,
                     DTTimeUnits::type unit ) const {

  DTLayerId detId( wheelId, stationId, sectorId,
                      slId,   layerId );
  int chanKey = detId.rawId();

  std::map<int,int>::iterator layer_iter = sortedLayers->find( chanKey );
  std::map<int,int>::iterator layer_iend = sortedLayers->end();

  if ( layer_iter == layer_iend ) return 1;

  int& lCode = layer_iter->second;
  int idprev = lCode / 1000;
  int length = lCode % 1000;
  int idnext = idprev + length;

  --idprev;
  int idtest;
  while ( ( length = idnext - idprev ) >= 2 ) {
    idtest = idprev + ( length / 2 );
    int cCell = dataList[idtest].first.cellId;
    if ( cCell < cellId ) {
      idprev = idtest;
      continue;
    }
    if ( cCell > cellId ) {
      idnext = idtest;
      continue;
    }
    idprev = idtest++;
    idnext = idtest;
    const DTT0Data& data( dataList[idprev].second );
    t0mean = data.t0mean;
    t0rms  = data.t0rms;
    if ( unit == DTTimeUnits::ns ) {
      t0mean *= nsPerCount;
      t0rms  *= nsPerCount;
    }
    return 0;
  }
  std::cout << "cell not found!" << std::endl;
  return 1;
}


int DTT0::get( const DTWireId& id,
               float& t0mean,
               float& t0rms,
               DTTimeUnits::type unit ) const {
  return get( id.wheel(),
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
  delete dBuf;
  dBuf = 0;
  dataList.clear();
  return;
}


int DTT0::set( int   wheelId,
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

  if ( dBuf == 0 ) cacheMap();

  DTWireId detId( wheelId, stationId, sectorId,
                  slId,   layerId,   cellId );
  int chanKey = detId.rawId();

  int ientry;
  std::map<int,int>::const_iterator buf_iter = dBuf->find( chanKey );
  std::map<int,int>::const_iterator buf_iend = dBuf->end();
  int searchStatus = ( buf_iter == buf_iend );

  if ( !searchStatus ) {
    ientry = buf_iter->second;
    DTT0Data& data( dataList[ientry].second );
    data.t0mean = t0mean;
    data.t0rms  = t0rms;
    return -1;
  }
  else {
    DTT0Id key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    key.     slId =      slId;
    key.  layerId =   layerId;
    key.   cellId =    cellId;
    DTT0Data data;
    data.t0mean = t0mean;
    data.t0rms  = t0rms;
    ientry = dataList.size();
    dataList.push_back( std::pair<const DTT0Id,DTT0Data>( key, data ) );
    dBuf->insert( std::pair<int,int>( detId.rawId(), ientry ) );
    return 0;
  }

  return 99;

}


int DTT0::set( const DTWireId& id,
               float t0mean,
               float t0rms,
               DTTimeUnits::type unit ) {
  return set( id.wheel(),
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


void DTT0::sortData() {
  std::cout << "test data" << std::endl;
  if ( sortedLayers != 0 ) return;
  if ( dBuf == 0 ) checkOrder();
  if ( sortedLayers != 0 ) return;
  if ( dBuf == 0 ) return;
  std::vector< std::pair<DTT0Id,DTT0Data> > tempList;
  std::cout << "sort data" << std::endl;
  std::map<int,int>::const_iterator iter = dBuf->begin();
  std::map<int,int>::const_iterator iend = dBuf->end();
  while ( iter != iend ) tempList.push_back( dataList[iter++->second] );
  dataList = tempList;
  delete dBuf;
  dBuf = 0;
  return;
}


DTT0::const_iterator DTT0::begin() const {
  return dataList.begin();
}


DTT0::const_iterator DTT0::end() const {
  return dataList.end();
}


std::string DTT0::mapName() const {
  std::stringstream name;
  name << dataVersion << "_map_T0" << this;
  return name.str();
}


bool DTT0::checkOrder() const {
  std::cout << "check order" << std::endl;
  delete sortedLayers;
  sortedLayers = new std::map<int,int>;
  int entryNum = 0;
  int entryMax = dataList.size();
  int oldId = 0;
  int lCell = -999999999;
  bool layerOrder = true;
  while ( entryNum < entryMax ) {
    const DTT0Id& chan = dataList[entryNum].first;
    DTLayerId detId( chan.  wheelId, chan.stationId, chan. sectorId,
                     chan.     slId, chan.  layerId );
    int rawId = detId.rawId();
    std::map<int,int>::iterator layer_iter = sortedLayers->find( rawId );
    std::map<int,int>::iterator layer_iend = sortedLayers->end();
    if ( layer_iter == layer_iend ) {
      sortedLayers->insert( std::pair<int,int>( rawId,
                                                1 + ( entryNum * 1000 ) ) );
      oldId = rawId;
    }
    else {
      int& lCode = layer_iter->second;
      int offset = lCode / 1000;
      int length = lCode % 1000;
      int ncells = entryNum - offset;
      if ( ( ncells != length     ) ||
           (  rawId != oldId      ) ||
           ( chan.cellId <= lCell ) ) layerOrder = false;
      layer_iter->second = ( offset * 1000 ) + ncells + 1;
    }
    lCell = chan.cellId;
    entryNum++;
  }
  if ( !layerOrder ) cacheMap();
  return layerOrder;
}


void DTT0::cacheMap() const {

  std::cout << "cache map" << std::endl;
  delete sortedLayers;
  sortedLayers = 0;
  dBuf = new std::map<int,int>;

  int entryNum = 0;
  int entryMax = dataList.size();
  while ( entryNum < entryMax ) {

    const DTT0Id& chan = dataList[entryNum].first;
    DTWireId detId( chan.  wheelId, chan.stationId, chan. sectorId,
                    chan.     slId, chan.  layerId, chan.   cellId );
    dBuf->insert( std::pair<int,int>( detId.rawId(), entryNum++ ) );

  }

  return;

}

