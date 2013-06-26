/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/20 18:20:08 $
 *  $Revision: 1.2 $
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
//#include "CondFormats/DTObjects/interface/DTDataBuffer.h"
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
DTLVStatus::DTLVStatus():
  dataVersion( " " ) {
  dataList.reserve( 10 );
  dBuf = 0;
}


DTLVStatus::DTLVStatus( const std::string& version ):
  dataVersion( version ) {
  dataList.reserve( 10 );
  dBuf = 0;
}


DTLVStatusId::DTLVStatusId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ) {
}


DTLVStatusData::DTLVStatusData() :
  flagCFE( 0 ),
  flagDFE( 0 ),
  flagCMC( 0 ),
  flagDMC( 0 ) {
}


//--------------
// Destructor --
//--------------
DTLVStatus::~DTLVStatus() {
//  DTDataBuffer<int,int>::dropBuffer( mapName() );
  delete dBuf;
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
                     int&  flagCFE,
                     int&  flagDFE,
                     int&  flagCMC,
                     int&  flagDMC ) const {
  flagCFE = 0;
  flagDFE = 0;
  flagCMC = 0;
  flagDMC = 0;

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
  chanKey.reserve(3);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );
  if ( !searchStatus ) {
    const DTLVStatusData& data( dataList[ientry].second );
    flagCFE = data.flagCFE;
    flagDFE = data.flagDFE;
    flagCMC = data.flagCMC;
    flagDMC = data.flagDMC;
  }

  return searchStatus;

}


int DTLVStatus::get( const DTChamberId& id,
                     int&  flagCFE,
                     int&  flagDFE,
                     int&  flagCMC,
                     int&  flagDMC ) const {
  return get( id.wheel(),
              id.station(),
              id.sector(),
              flagCFE,
              flagDFE,
              flagCMC,
              flagDMC );
}


const
std::string& DTLVStatus::version() const {
  return dataVersion;
}


std::string& DTLVStatus::version() {
  return dataVersion;
}


void DTLVStatus::clear() {
//  DTDataBuffer<int,int>::dropBuffer( mapName() );
  delete dBuf;
  dBuf = 0;
  dataList.clear();
  return;
}


int DTLVStatus::set( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int   flagCFE,
                     int   flagDFE,
                     int   flagCMC,
                     int   flagDMC ) {

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
  chanKey.reserve(3);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  int ientry;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), ientry );

  if ( !searchStatus ) {
    DTLVStatusData& data( dataList[ientry].second );
    data.flagCFE = flagCFE;
    data.flagDFE = flagDFE;
    data.flagCMC = flagCMC;
    data.flagDMC = flagDMC;
    return -1;
  }
  else {
    DTLVStatusId key;
    key.  wheelId =   wheelId;
    key.stationId = stationId;
    key. sectorId =  sectorId;
    DTLVStatusData data;
    data.flagCFE = flagCFE;
    data.flagDFE = flagDFE;
    data.flagCMC = flagCMC;
    data.flagDMC = flagDMC;
    ientry = dataList.size();
    dataList.push_back( std::pair<DTLVStatusId,
                                  DTLVStatusData>( key, data ) );
    dBuf->insert( chanKey.begin(), chanKey.end(), ientry );
    return 0;
  }

  return 99;

}


int DTLVStatus::set( const DTChamberId& id,
                     int   flagCFE,
                     int   flagDFE,
                     int   flagCMC,
                     int   flagDMC  ) {
  return set( id.wheel(),
              id.station(),
              id.sector(),
              flagCFE,
              flagDFE,
              flagCMC,
              flagDMC );
}


int DTLVStatus::setFlagCFE( int   wheelId,
                            int stationId,
                            int  sectorId,
                            int      flag ) {
  int   flagCFE;
  int   flagDFE;
  int   flagCMC;
  int   flagDMC;
  get(   wheelId,
       stationId,
        sectorId,
         flagCFE,
         flagDFE,
         flagCMC,
         flagDMC );
  return set(   wheelId,
              stationId,
               sectorId,
                   flag,
                flagDFE,
                flagCMC,
                flagDMC );
}


int DTLVStatus::setFlagCFE( const DTChamberId& id,
                            int   flag ) {
  return setFlagCFE( id.wheel(),
                     id.station(),
                     id.sector(),
                     flag );
}


int DTLVStatus::setFlagDFE( int   wheelId,
                            int stationId,
                            int  sectorId,
                            int      flag ) {
  int   flagCFE;
  int   flagDFE;
  int   flagCMC;
  int   flagDMC;
  get(   wheelId,
       stationId,
        sectorId,
         flagCFE,
         flagDFE,
         flagCMC,
         flagDMC );
  return set(   wheelId,
              stationId,
               sectorId,
                flagCFE,
                   flag,
                flagCMC,
                flagDMC );
}


int DTLVStatus::setFlagDFE( const DTChamberId& id,
                            int   flag ) {
  return setFlagDFE( id.wheel(),
                     id.station(),
                     id.sector(),
                     flag );
}


int DTLVStatus::setFlagCMC( int   wheelId,
                            int stationId,
                            int  sectorId,
                            int      flag ) {
  int   flagCFE;
  int   flagDFE;
  int   flagCMC;
  int   flagDMC;
  get(   wheelId,
       stationId,
        sectorId,
         flagCFE,
         flagDFE,
         flagCMC,
         flagDMC );
  return set(   wheelId,
              stationId,
               sectorId,
                flagCFE,
                flagDFE,
                   flag,
                flagDMC );
}


int DTLVStatus::setFlagCMC( const DTChamberId& id,
                            int   flag ) {
  return setFlagCMC( id.wheel(),
                     id.station(),
                     id.sector(),
                     flag );
}


int DTLVStatus::setFlagDMC( int   wheelId,
                            int stationId,
                            int  sectorId,
                            int      flag ) {
  int   flagCFE;
  int   flagDFE;
  int   flagCMC;
  int   flagDMC;
  get(   wheelId,
       stationId,
        sectorId,
         flagCFE,
         flagDFE,
         flagCMC,
         flagDMC );
  return set(   wheelId,
              stationId,
               sectorId,
                flagCFE,
                flagDFE,
                flagCMC,
                   flag );
}


int DTLVStatus::setFlagDMC( const DTChamberId& id,
                            int   flag ) {
  return setFlagDMC( id.wheel(),
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

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::openBuffer( mName );
  DTBufferTree<int,int>** pBuf;
  pBuf = const_cast<DTBufferTree<int,int>**>( &dBuf );
  *pBuf = new DTBufferTree<int,int>;

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

