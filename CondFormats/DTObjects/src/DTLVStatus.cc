/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/20 12:00:00 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTLVStatus.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

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
DTLVStatus::DTLVStatus():
  dataVersion( " " ) {
  dataList.reserve( 10 );
}


DTLVStatus::DTLVStatus( const std::string& version ):
  dataVersion( version ) {
  dataList.reserve( 10 );
}


DTLVStatusId::DTLVStatusId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ) {
}


DTLVStatusData::DTLVStatusData() :
  flag( 0 ) {
}


//--------------
// Destructor --
//--------------
DTLVStatus::~DTLVStatus() {
  DTDataBuffer<int,int>::dropBuffer( mapName() );
}


DTLVStatusId::~DTLVStatusId() {
}


DTLVStatusData::~DTLVStatusData() {
}


//--------------
// Operations --
//--------------
int DTLVStatus::get( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int&     flag ) const {
  flag = 0;

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::findBuffer( mName );
  if ( dBuf == 0 ) {
    cacheMap();
    dBuf =
    DTDataBuffer<int,int>::findBuffer( mName );
  }

  std::vector<int> chanKey;
  chanKey.reserve(3);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );
  if ( !searchStatus ) {
    const DTLVStatusData& data( dataList[ientry].second );
    flag = data.flag;
  }

  return searchStatus;

}


int DTLVStatus::get( const DTChamberId& id,
                      int&  flag ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              flag );
}


const
std::string& DTLVStatus::version() const {
  return dataVersion;
}


std::string& DTLVStatus::version() {
  return dataVersion;
}


void DTLVStatus::clear() {
  DTDataBuffer<int,int>::dropBuffer( mapName() );
  dataList.clear();
  return;
}


int DTLVStatus::set( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      flag ) {

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::findBuffer( mName );
  if ( dBuf == 0 ) {
    cacheMap();
    dBuf =
    DTDataBuffer<int,int>::findBuffer( mName );
  }
  std::vector<int> chanKey;
  chanKey.reserve(3);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );

  if ( !searchStatus ) {
    DTLVStatusData& data( dataList[ientry].second );
    data.flag = flag;
    return -1;
  }
  else {
    DTLVStatusId key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    DTLVStatusData data;
    data.flag = flag;
    ientry = dataList.size();
    dataList.push_back( std::pair<DTLVStatusId,
                                  DTLVStatusData>( key, data ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
    return 0;
  }

  return 99;

}


int DTLVStatus::set( const DTChamberId& id,
                      int flag  ) {
  return set( id.wheel(),
              id.station(),
              id.sector(),
              flag );
}


DTLVStatus::const_iterator DTLVStatus::begin() const {
  return dataList.begin();
}


DTLVStatus::const_iterator DTLVStatus::end() const {
  return dataList.end();
}


std::string DTLVStatus::mapName() const {
  std::stringstream name;
  name << dataVersion << "_map_Mtime" << this;
  return name.str();
}


void DTLVStatus::cacheMap() const {

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::openBuffer( mName );

  int entryNum = 0;
  int entryMax = dataList.size();
  std::vector<int> chanKey;
  chanKey.reserve(3);
  while ( entryNum < entryMax ) {

    const DTLVStatusId& chan = dataList[entryNum].first;

    chanKey.clear();
    chanKey.push_back( chan.  wheelId );
    chanKey.push_back( chan.stationId );
    chanKey.push_back( chan. sectorId );
    dBuf->insert( chanKey.begin(), chanKey.end(), entryNum++ );

  }

  return;

}

