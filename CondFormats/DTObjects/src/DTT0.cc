/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/05/06 14:41:53 $
 *  $Revision: 1.21 $
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
#include "CondFormats/DTObjects/interface/DTSequentialLayerNumber.h"

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
//  dBuf = sortedLayers = 0;
  dBuf = 0;
  sortedLayers = 0;
  sequencePtr = 0;
}


DTT0::DTT0( const std::string& version ):
  dataVersion( version ),
  nsPerCount( 25.0 / 32.0 ) {
  dataList.reserve( 12000 );
//  dBuf = sortedLayers = 0;
  dBuf = 0;
  sortedLayers = 0;
  sequencePtr = 0;
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
  delete sequencePtr;
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
//  if ( sortedLayers != 0 ) std::cout << "sorted data" << std::endl;
//  else                     std::cout << "random data" << std::endl;
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

//  DTWireId detId( wheelId, stationId, sectorId,
//                  slId,   layerId,   cellId );
//  int chanKey = detId.rawId();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  chanKey.push_back(      slId );
  chanKey.push_back(   layerId );
  chanKey.push_back(    cellId );

  int ientry;
//  std::map<int,int>::const_iterator buf_iter = dBuf->find( chanKey );
//  std::map<int,int>::const_iterator buf_iend = dBuf->end();
//  int searchStatus = ( buf_iter == buf_iend );
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );
  if ( !searchStatus ) {
//    ientry = buf_iter->second;
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


int DTT0::maxCellsPerLayer() const {
  return 2000;
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

  int lCode;
  int seqId;
  std::vector<int>& sequenceLay = *sequencePtr;

  if ( ( seqId = DTSequentialLayerNumber::id( wheelId, stationId, sectorId,
                                              slId,   layerId ) ) < 0 ) {
//  DTLayerId detId( wheelId, stationId, sectorId,
//                      slId,   layerId );
//  int chanKey = detId.rawId();
    std::vector<int> chanKey;
    chanKey.reserve(6);
    chanKey.push_back(   wheelId );
    chanKey.push_back( stationId );
    chanKey.push_back(  sectorId );
    chanKey.push_back(      slId );
    chanKey.push_back(   layerId );

//  std::map<int,int>::iterator layer_iter = sortedLayers->find( chanKey );
//  std::map<int,int>::iterator layer_iend = sortedLayers->end();
//
//  if ( layer_iter == layer_iend ) return 1;
//
//  int& lCode = layer_iter->second;
    if ( sortedLayers->find( chanKey.begin(), chanKey.end(), lCode ) )
         return 1;
  }
  else {
    lCode = sequenceLay[seqId];
  }
  int mCells = maxCellsPerLayer();
  int idprev = lCode / mCells;
  int length = lCode % mCells;
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
  // std::cout << "cell not found!" << std::endl;
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

//  DTWireId detId( wheelId, stationId, sectorId,
//                  slId,   layerId,   cellId );
//  int chanKey = detId.rawId();
  if ( dBuf == 0 ) cacheMap();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  chanKey.push_back(      slId );
  chanKey.push_back(   layerId );
  chanKey.push_back(    cellId );

  int ientry;
//  std::map<int,int>::const_iterator buf_iter = dBuf->find( chanKey );
//  std::map<int,int>::const_iterator buf_iend = dBuf->end();
//  int searchStatus = ( buf_iter == buf_iend );
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );

  if ( !searchStatus ) {
//    ientry = buf_iter->second;
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
//    dBuf->insert( std::pair<int,int>( detId.rawId(), ientry ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
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
  if ( sortedLayers != 0 ) return;
  if ( dBuf == 0 ) checkOrder();
  if ( sortedLayers != 0 ) return;
  if ( dBuf == 0 ) return;
  std::vector< std::pair<DTT0Id,DTT0Data> > tempList;
//  std::map<int,int>::const_iterator iter = dBuf->begin();
//  std::map<int,int>::const_iterator iend = dBuf->end();
//  while ( iter != iend ) tempList.push_back( dataList[iter++->second] );
  std::vector<int> indexList = dBuf->contList();
  std::vector<int>::const_iterator iter = indexList.begin();
  std::vector<int>::const_iterator iend = indexList.end();
  while ( iter != iend ) tempList.push_back( dataList[*iter++] );
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

  delete sortedLayers;
  delete sequencePtr;
//  sortedLayers = new std::map<int,int>;
  sortedLayers = new DTBufferTree<int,int>;
//  sequencePtr = new std::vector<int>;
//  sequencePtr->reserve( DTSequentialLayerNumber::max() + 2 );
  sequencePtr = new std::vector<int>( DTSequentialLayerNumber::max() + 2 );
  std::vector<int>::iterator iter = sequencePtr->begin();
  std::vector<int>::iterator iend = sequencePtr->end();
  while ( iter != iend ) *iter++ = 0;

  int entryNum = 0;
  int mCells = maxCellsPerLayer();
  int entryMax = dataList.size();
//  int oldId = 0;
  std::vector<int> chanOld;
  int lCell = -999999999;
  bool layerOrder = true;

  int lCode;
  int seqId;
  int oldId = 0;
  std::vector<int>& sequenceLay = *sequencePtr;

  std::vector<int> chanKey;
  chanKey.reserve(6);

  while ( entryNum < entryMax ) {
    const DTT0Id& chan = dataList[entryNum].first;
    if ( ( seqId = DTSequentialLayerNumber::id( chan.  wheelId,
                                                chan.stationId,
                                                chan. sectorId,
                                                chan.     slId,
                                                chan.  layerId ) ) < 0 ) {
      oldId = 0;
//      std::cout << "out of sequence" << std::endl;
//    DTLayerId detId( chan.  wheelId, chan.stationId, chan. sectorId,
//                     chan.     slId, chan.  layerId );
//    int rawId = detId.rawId();
      chanKey.clear();
      chanKey.push_back( chan.  wheelId );
      chanKey.push_back( chan.stationId );
      chanKey.push_back( chan. sectorId );
      chanKey.push_back( chan.     slId );
      chanKey.push_back( chan.  layerId );
//    std::map<int,int>::iterator layer_iter = sortedLayers->find( rawId );
//    std::map<int,int>::iterator layer_iend = sortedLayers->end();
//    if ( layer_iter == layer_iend ) {
      if ( sortedLayers->find( chanKey.begin(), chanKey.end(), lCode ) ) {
//      sortedLayers->insert( std::pair<int,int>( rawId,
//                                                1 + ( entryNum * 1000 ) ) );
//      oldId = rawId;
        sortedLayers->insert( chanKey.begin(), chanKey.end(),
                              1 + ( entryNum * mCells ) );
        chanOld = chanKey;
      }
      else {
//      int& lCode = layer_iter->second;
        int offset = lCode / mCells;
        int length = lCode % mCells;
        int ncells = entryNum - offset;
//      if ( ( ncells != length     ) ||
//           (  rawId != oldId      ) ||
//           ( chan.cellId <= lCell ) ) layerOrder = false;
//      layer_iter->second = ( offset * 1000 ) + ncells + 1;
        if ( ( ncells      >= mCells     ) ||
             ( ncells      != length     ) ||
             ( chanKey[0]  != chanOld[0] ) ||
             ( chanKey[1]  != chanOld[1] ) ||
             ( chanKey[2]  != chanOld[2] ) ||
             ( chanKey[3]  != chanOld[3] ) ||
             ( chanKey[4]  != chanOld[4] ) ||
             ( chan.cellId <= lCell      ) ) layerOrder = false;
        sortedLayers->insert( chanKey.begin(), chanKey.end(),
                              ( offset * mCells ) + ncells + 1 );
      }
    }
    else {
      chanOld.clear();
//      std::cout << "inside sequence" << std::endl;
      lCode = sequenceLay[seqId];
      if ( lCode == 0 ) {
        sequenceLay[seqId] = 1 + ( entryNum * mCells );
        oldId = seqId;
      }
      else {
        int offset = lCode / mCells;
        int length = lCode % mCells;
        int ncells = entryNum - offset;
        if ( ( ncells      >= mCells     ) ||
             ( ncells      != length     ) ||
             ( seqId       != oldId      ) ||
             ( chan.cellId <= lCell      ) ) layerOrder = false;
        sequenceLay[seqId] = ( offset * mCells ) + ncells + 1;
      }
    }
    lCell = chan.cellId;
    entryNum++;
  }
  if ( !layerOrder ) cacheMap();
  return layerOrder;
}


void DTT0::cacheMap() const {

  delete sortedLayers;
  sortedLayers = 0;
//  dBuf = new std::map<int,int>;
  dBuf = new DTBufferTree<int,int>;

  int entryNum = 0;
  int entryMax = dataList.size();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  while ( entryNum < entryMax ) {

   const DTT0Id& chan = dataList[entryNum].first;

//    DTWireId detId( chan.  wheelId, chan.stationId, chan. sectorId,
//                    chan.     slId, chan.  layerId, chan.   cellId );
//    dBuf->insert( std::pair<int,int>( detId.rawId(), entryNum++ ) );
    chanKey.clear();
    chanKey.push_back( chan.  wheelId );
    chanKey.push_back( chan.stationId );
    chanKey.push_back( chan. sectorId );
    chanKey.push_back( chan.     slId );
    chanKey.push_back( chan.  layerId );
    chanKey.push_back( chan.   cellId );
    dBuf->insert( chanKey.begin(), chanKey.end(), entryNum++ );

  }

  return;

}

