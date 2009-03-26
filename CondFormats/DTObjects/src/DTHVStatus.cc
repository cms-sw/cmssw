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
#include "CondFormats/DTObjects/interface/DTHVStatus.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTDataBuffer.h"
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
}


DTHVStatus::DTHVStatus( const std::string& version ):
  dataVersion( version ) {
  dataList.reserve( 10 );
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
  DTDataBuffer<int,int>::dropBuffer( mapName() );
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

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::findBuffer( mName );
  if ( dBuf == 0 ) {
    cacheMap();
    dBuf =
    DTDataBuffer<int,int>::findBuffer( mName );
  }

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


//int DTHVStatus::offChannelsNumber( const DTChamberId& id ) const {
//  return 0;
//}


const
std::string& DTHVStatus::version() const {
  return dataVersion;
}


std::string& DTHVStatus::version() {
  return dataVersion;
}


void DTHVStatus::clear() {
  DTDataBuffer<int,int>::dropBuffer( mapName() );
  dataList.clear();
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

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::findBuffer( mName );
  if ( dBuf == 0 ) {
    cacheMap();
    dBuf =
    DTDataBuffer<int,int>::findBuffer( mName );
  }
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

  std::string mName = mapName();
  DTBufferTree<int,int>* dBuf =
  DTDataBuffer<int,int>::openBuffer( mName );

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


