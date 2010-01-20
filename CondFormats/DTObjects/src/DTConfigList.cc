/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/07/15 15:57:23 $
 *  $Revision: 1.4 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondFormats/DTObjects/interface/DTConfigList.h"

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
DTConfigList::DTConfigList():
  dataVersion( " " ) {
  dBuf = 0;
}


DTConfigList::DTConfigList( const std::string& version ):
  dataVersion( version ) {
  dBuf = 0;
}


DTConfigToken::DTConfigToken() :
  id( 0 ),
  ref( " " ) {
}


//--------------
// Destructor --
//--------------
DTConfigList::~DTConfigList() {
//  DTDataBuffer<int,int>::dropBuffer( mapName() );
  delete dBuf;
}


DTConfigToken::~DTConfigToken() {
}


//--------------
// Operations --
//--------------
int DTConfigList::get( int id, DTConfigToken& token ) const {

  token.id = 0;
  token.ref = " ";

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer<int,int>::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();

  std::vector<int> cfgKey;
  cfgKey.push_back( id );
  int ientry;
  int searchStatus = dBuf->find( cfgKey.begin(), cfgKey.end(), ientry );
  if ( !searchStatus ) token = dataList[ientry].second;

  return searchStatus;

}


const
std::string& DTConfigList::version() const {
  return dataVersion;
}


std::string& DTConfigList::version() {
  return dataVersion;
}


void DTConfigList::clear() {
//  DTDataBuffer<int,int>::dropBuffer( mapName() );
  delete dBuf;
  dBuf = 0;
  dataList.clear();
  return;
}


int DTConfigList::set( int id, const DTConfigToken& token ) {

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer<int,int>::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();
  std::vector<int> cfgKey;
  cfgKey.push_back( id );
  int ientry;
  int searchStatus = dBuf->find( cfgKey.begin(), cfgKey.end(), ientry );
  if ( !searchStatus ) {
    dataList[ientry].second = token;
    return -1;
  }
  else {
    ientry = dataList.size();
    dataList.push_back( std::pair<int,DTConfigToken>( id, token ) );
    dBuf->insert( cfgKey.begin(), cfgKey.end(), ientry );
    return 0;
  }

}


DTConfigList::const_iterator DTConfigList::begin() const {
  return dataList.begin();
}


DTConfigList::const_iterator DTConfigList::end() const {
  return dataList.end();
}


std::string DTConfigList::mapName() const {
  std::stringstream name;
  name << dataVersion << "_map_config_bricks" << this;
  return name.str();
}


void DTConfigList::cacheMap() const {

//  std::string mName = mapName();
//  DTBufferTree<int,int>* dBuf =
//  DTDataBuffer<int,int>::openBuffer( mName );

  DTBufferTree<int,int>** pBuf;
  pBuf = const_cast<DTBufferTree<int,int>**>( &dBuf );
  *pBuf = new DTBufferTree<int,int>;

  int entryNum = 0;
  int entryMax = dataList.size();
  while ( entryNum < entryMax ) {

    int cfgId = dataList[entryNum].first;

    std::vector<int> cfgKey;
    cfgKey.push_back( cfgId );
    dBuf->insert( cfgKey.begin(), cfgKey.end(), entryNum++ );

  }

  return;

}

