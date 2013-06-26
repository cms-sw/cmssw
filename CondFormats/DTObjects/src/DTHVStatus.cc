/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/20 18:20:08 $
 *  $Revision: 1.3 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTHVStatus.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "CondFormats/DTObjects/interface/DTDataBuffer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

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
DTHVStatus::DTHVStatus():
  dataVersion( " " ) {
  dataList.reserve( 10 );
  dBuf = 0;
}


DTHVStatus::DTHVStatus( const std::string& version ):
  dataVersion( version ) {
  dataList.reserve( 10 );
  dBuf = 0;
}


DTHVStatusId::DTHVStatusId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     partId( 0 ) {
}


DTHVStatusData::DTHVStatusData() :
  flagA( 0 ),
  flagC( 0 ),
  flagS( 0 ) {
}


//--------------
// Destructor --
//--------------
DTHVStatus::~DTHVStatus() {
//  DTDataBuffer<int,int>::dropBuffer( mapName() );
  delete dBuf;
}


DTHVStatusId::~DTHVStatusId() {
}


DTHVStatusData::~DTHVStatusData() {
}


//--------------
// Operations --
//--------------
int DTHVStatus::get( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    partId,
                     int&    fCell,
                     int&    lCell,
                     int&    flagA,
                     int&    flagC,
                     int&    flagS ) const {
  fCell =
  lCell =
  flagA =
  flagC = 
  flagS = 0;

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer<int,int>::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();

  std::vector<int> chanKey;
  chanKey.reserve(6);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  chanKey.push_back(      slId );
  chanKey.push_back(   layerId );
  chanKey.push_back(    partId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );
  if ( !searchStatus ) {
    const DTHVStatusData& data( dataList[ientry].second );
    fCell = data.fCell;
    lCell = data.lCell;
    flagA = data.flagA;
    flagC = data.flagC;
    flagS = data.flagS;
  }

  return searchStatus;

}


int DTHVStatus::get( const DTLayerId& id,
                     int    partId,
                     int&    fCell,
                     int&    lCell,
                     int&    flagA,
                     int&    flagC,
                     int&    flagS ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(),
              id.layer(),
              partId,
              fCell, lCell, flagA, flagC, flagS );
}


int DTHVStatus::get( const DTWireId& id,
                     int&         flagA,
                     int&         flagC,
                     int&         flagS ) const {
  flagA = flagC = flagS = 0;
  int iCell = id.wire();
  int fCell;
  int lCell;
  int
  fCheck = get( id.wheel(),
                id.station(),
                id.sector(),
                id.superLayer(),
                id.layer(),
                0,     fCell, lCell,
                flagA, flagC, flagS );
  if ( ( fCheck == 0 ) &&
       ( fCell <= iCell ) &&
       ( lCell >= iCell ) ) return 0;
  fCheck = get( id.wheel(),
                id.station(),
                id.sector(),
                id.superLayer(),
                id.layer(),
                1,     fCell, lCell,
                flagA, flagC, flagS );
  if ( ( fCheck == 0 ) &&
       ( fCell <= iCell ) &&
       ( lCell >= iCell ) ) return 0;
  flagA = flagC = flagS = 0;
  return 1;
}


int DTHVStatus::offChannelsNumber() const {
  int offNum = 0;
  DTHVStatus::const_iterator iter = begin();
  DTHVStatus::const_iterator iend = end();
  while ( iter != iend ) {
    const std::pair<DTHVStatusId,DTHVStatusData>& entry = *iter++;
    DTHVStatusId   hvId = entry.first;
    DTHVStatusData data = entry.second;
    int offA = data.flagA & 1;
    int offC = data.flagC & 1;
    int offS = data.flagS & 1;
    if ( offA || offC || offS )
         offNum += ( 1 + data.lCell - data.fCell );
  }
  return offNum;
}


int DTHVStatus::offChannelsNumber( const DTChamberId& id ) const {
  int offNum = 0;
  DTHVStatus::const_iterator iter = begin();
  DTHVStatus::const_iterator iend = end();
  while ( iter != iend ) {
    const std::pair<DTHVStatusId,DTHVStatusData>& entry = *iter++;
    DTHVStatusId   hvId = entry.first;
    DTHVStatusData data = entry.second;
    if ( hvId.  wheelId != id.  wheel() ) continue;
    if ( hvId.stationId != id.station() ) continue;
    if ( hvId. sectorId != id. sector() ) continue;
    int offA = data.flagA & 1;
    int offC = data.flagC & 1;
    int offS = data.flagS & 1;
    if ( offA || offC || offS )
         offNum += ( 1 + data.lCell - data.fCell );
  }
  return offNum;
}


int DTHVStatus::badChannelsNumber() const {
  int offNum = 0;
  DTHVStatus::const_iterator iter = begin();
  DTHVStatus::const_iterator iend = end();
  while ( iter != iend ) {
    const std::pair<DTHVStatusId,DTHVStatusData>& entry = *iter++;
    DTHVStatusId   hvId = entry.first;
    DTHVStatusData data = entry.second;
    if ( data.flagA || data.flagC || data.flagS )
         offNum += ( 1 + data.lCell - data.fCell );
  }
  return offNum;
}


int DTHVStatus::badChannelsNumber( const DTChamberId& id ) const {
  int offNum = 0;
  DTHVStatus::const_iterator iter = begin();
  DTHVStatus::const_iterator iend = end();
  while ( iter != iend ) {
    const std::pair<DTHVStatusId,DTHVStatusData>& entry = *iter++;
    DTHVStatusId   hvId = entry.first;
    DTHVStatusData data = entry.second;
    if ( hvId.  wheelId != id.  wheel() ) continue;
    if ( hvId.stationId != id.station() ) continue;
    if ( hvId. sectorId != id. sector() ) continue;
    if ( data.flagA || data.flagC || data.flagS )
         offNum += ( 1 + data.lCell - data.fCell );
  }
  return offNum;
}


const
std::string& DTHVStatus::version() const {
  return dataVersion;
}


std::string& DTHVStatus::version() {
  return dataVersion;
}


void DTHVStatus::clear() {
//  DTDataBuffer<int,int>::dropBuffer( mapName() );
  delete dBuf;
  dBuf = 0;
  dataList.clear();
  dataList.reserve( 10 );
  return;
}


int DTHVStatus::set( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    partId,
                     int     fCell,
                     int     lCell,
                     int     flagA,
                     int     flagC,
                     int     flagS ) {

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer<int,int>::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  chanKey.push_back(      slId );
  chanKey.push_back(   layerId );
  chanKey.push_back(    partId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );

  if ( !searchStatus ) {
    DTHVStatusData& data( dataList[ientry].second );
    data.fCell = fCell;
    data.lCell = lCell;
    data.flagA = flagA;
    data.flagC = flagC;
    data.flagS = flagS;
    return -1;
  }
  else {
    DTHVStatusId key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    key.     slId =      slId;
    key.  layerId =   layerId;
    key.   partId =    partId;
    DTHVStatusData data;
    data.fCell = fCell;
    data.lCell = lCell;
    data.flagA = flagA;
    data.flagC = flagC;
    data.flagS = flagS;
    ientry = dataList.size();
    dataList.push_back( std::pair<DTHVStatusId,
                                  DTHVStatusData>( key, data ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
    return 0;
  }

  return 99;

}


int DTHVStatus::set( const DTLayerId& id,
                     int    partId,
                     int     fCell,
                     int     lCell,
                     int     flagA,
                     int     flagC,
                     int     flagS ) {
  return set( id.wheel(),
              id.station(),
              id.sector(),
              id.superLayer(),
              id.layer(),
              partId,
              fCell, lCell, flagA, flagC, flagS );
}


int DTHVStatus::setFlagA( int   wheelId,
                          int stationId,
                          int  sectorId,
                          int      slId,
                          int   layerId,
                          int    partId,
                          int      flag ) {
  int fCell;
  int lCell;
  int flagA;
  int flagC;
  int flagS;
  get(   wheelId,
       stationId,
        sectorId,
            slId,
         layerId,
          partId,
           fCell,
           lCell,
           flagA,
           flagC,
           flagS );
  return set(   wheelId,
              stationId,
               sectorId,
                   slId,
                layerId,
                 partId,
                  fCell,
                  lCell,
                   flag,
                  flagC,
                  flagS );
}


int DTHVStatus::setFlagA( const DTLayerId& id,
                          int    partId,
                          int      flag ) {
  return setFlagA( id.wheel(),
                   id.station(),
                   id.sector(),
                   id.superLayer(),
                   id.layer(),
                   partId,
                   flag );
}


int DTHVStatus::setFlagC( int   wheelId,
                          int stationId,
                          int  sectorId,
                          int      slId,
                          int   layerId,
                          int    partId,
                          int      flag ) {
  int fCell;
  int lCell;
  int flagA;
  int flagC;
  int flagS;
  get(   wheelId,
       stationId,
        sectorId,
            slId,
         layerId,
          partId,
           fCell,
           lCell,
           flagA,
           flagC,
           flagS );
  return set(   wheelId,
              stationId,
               sectorId,
               slId,
                layerId,
                 partId,
                  fCell,
                  lCell,
                  flagA,
                   flag,
                  flagS );
}


int DTHVStatus::setFlagC( const DTLayerId& id,
                          int    partId,
                          int      flag ) {
  return setFlagC( id.wheel(),
                   id.station(),
                   id.sector(),
                   id.superLayer(),
                   id.layer(),
                   partId,
                   flag );
}


int DTHVStatus::setFlagS( int   wheelId,
                          int stationId,
                          int  sectorId,
                          int      slId,
                          int   layerId,
                          int    partId,
                          int      flag ) {
  int fCell;
  int lCell;
  int flagA;
  int flagC;
  int flagS;
  get(   wheelId,
       stationId,
        sectorId,
            slId,
         layerId,
          partId,
           fCell,
           lCell,
           flagA,
           flagC,
           flagS );
  return set(   wheelId,
              stationId,
               sectorId,
                   slId,
                layerId,
                 partId,
                  fCell,
                  lCell,
                  flagA,
                  flagC,
                   flag );
}


int DTHVStatus::setFlagS( const DTLayerId& id,
                          int    partId,
                          int      flag ) {
  return setFlagS( id.wheel(),
                   id.station(),
                   id.sector(),
                   id.superLayer(),
                   id.layer(),
                   partId,
                   flag );
}


DTHVStatus::const_iterator DTHVStatus::begin() const {
  return dataList.begin();
}


DTHVStatus::const_iterator DTHVStatus::end() const {
  return dataList.end();
}


std::string DTHVStatus::mapName() const {
  std::stringstream name;
  name << dataVersion << "_map_HV" << this;
  return name.str();
}


void DTHVStatus::cacheMap() const {

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::openBuffer( mName );
  DTBufferTree<int,int>** pBuf;
  pBuf = const_cast<DTBufferTree<int,int>**>( &dBuf );
  *pBuf = new DTBufferTree<int,int>;

  int entryNum = 0;
  int entryMax = dataList.size();
  std::vector<int> chanKey;
  chanKey.reserve(6);
  while ( entryNum < entryMax ) {

    const DTHVStatusId& chan = dataList[entryNum].first;

    chanKey.clear();
    chanKey.push_back( chan.  wheelId );
    chanKey.push_back( chan.stationId );
    chanKey.push_back( chan. sectorId );
    chanKey.push_back( chan.     slId );
    chanKey.push_back( chan.  layerId );
    chanKey.push_back( chan.   partId );
    dBuf->insert( chanKey.begin(), chanKey.end(), entryNum++ );

  }

  return;

}


