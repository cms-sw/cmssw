/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/05/23 15:59:44 $
 *  $Revision: 1.10 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <sstream>
#include <vector>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTCCBConfig::DTCCBConfig():
  dataVersion( " " ) {
  dataList.reserve( 1000 );
  dBuf = 0;
}


DTCCBConfig::DTCCBConfig( const std::string& version ):
  dataVersion( version ) {
  dataList.reserve( 1000 );
  dBuf = 0;
}


DTCCBId::DTCCBId() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ) {
}


DTConfigKey::DTConfigKey() :
  confType( 0 ),
  confKey ( 0 ) {
}


//--------------
// Destructor --
//--------------
DTCCBConfig::~DTCCBConfig() {
  resetMap();
//  DTDataBuffer< int,std::vector<int>* >::dropBuffer( mapName() );
  delete dBuf;
}


DTCCBId::~DTCCBId() {
}


DTConfigKey::~DTConfigKey() {
}


//--------------
// Operations --
//--------------
std::vector<DTConfigKey> DTCCBConfig::fullKey() const {
  return fullConfigKey;
}


int DTCCBConfig::stamp() const {
  return timeStamp;
}


int DTCCBConfig::configKey( int   wheelId,
                            int stationId,
                            int  sectorId,
                            std::vector<int>& confKey ) const {

  confKey.clear();
//  std::string mName = mapName();
//  DTBufferTree< int,std::vector<int>* >* dBuf =
//  DTDataBuffer< int,std::vector<int>* >::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer< int,std::vector<int>* >::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();

  std::vector<int> chanKey;
  chanKey.reserve(3);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );
  std::vector<int>* confPtr;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), confPtr );
  if ( !searchStatus ) confKey = *confPtr;

  return searchStatus;

}


int DTCCBConfig::configKey( const DTChamberId& id,
                            std::vector<int>& confKey ) const {
  return configKey( id.wheel(),
                    id.station(),
                    id.sector(),
                    confKey );
}


DTCCBConfig::ccb_config_map
DTCCBConfig::configKeyMap() const {

  ccb_config_map keyList;
  std::vector< std::pair<DTCCBId,int>* > tempList;
  const_iterator d_iter = begin();
  const_iterator d_iend = end();
  while ( d_iter != d_iend ) tempList.push_back(
                             new std::pair<DTCCBId,int>( *d_iter++ ) );
  std::vector< std::pair<DTCCBId,int>* >::iterator t_iter = tempList.begin();
  std::vector< std::pair<DTCCBId,int>* >::iterator t_iend = tempList.end();
  while ( t_iter != t_iend ) {
    std::pair<DTCCBId,int>* ptr = *t_iter++;
    if ( ptr == 0 ) continue;
    DTCCBId& ccbId = ptr->first;
    std::vector<int> cfgKeys;
    cfgKeys.push_back( ptr->second );
    std::vector< std::pair<DTCCBId,int>* >::iterator n_iter( t_iter );
    while( n_iter != t_iend ) {
      std::pair<DTCCBId,int>*& pck = *n_iter++;
      if ( pck == 0 ) continue; 
      DTCCBId& chkId = pck->first;
      if ( ( chkId.  wheelId == ccbId.  wheelId ) && 
           ( chkId.stationId == ccbId.stationId ) && 
           ( chkId. sectorId == ccbId. sectorId ) ) {
        cfgKeys.push_back( pck->second );
        delete pck;
        pck = 0;
      }
    }
    keyList.push_back( std::pair< DTCCBId,std::vector<int> >( ccbId,
                                                              cfgKeys ) );
    delete ptr;
  }
  return keyList;

}


const
std::string& DTCCBConfig::version() const {
  return dataVersion;
}


std::string& DTCCBConfig::version() {
  return dataVersion;
}


void DTCCBConfig::clear() {
  resetMap();
//  DTDataBuffer< int,std::vector<int>* >::dropBuffer( mapName() );
  delete dBuf;
  dBuf = 0;
  dataList.clear();
  return;
}


void DTCCBConfig::setFullKey( const std::vector<DTConfigKey>& key ) {
  fullConfigKey = key;
}


void DTCCBConfig::setStamp( int s ) {
  timeStamp = s;
}


int DTCCBConfig::setConfigKey( int   wheelId,
                               int stationId,
                               int  sectorId,
                               const std::vector<int>& confKey ) {

//  std::string mName = mapName();
//  DTBufferTree< int,std::vector<int>* >* dBuf =
//  DTDataBuffer< int,std::vector<int>* >::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer< int,std::vector<int>* >::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();

  std::vector<int> chanKey;
  chanKey.reserve(3);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );

  std::vector<int>* confPtr;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), confPtr );

  if ( !searchStatus ) {
    std::vector< std::pair<DTCCBId,int> > tempList;
    const_iterator iter = dataList.begin();
    const_iterator iend = dataList.end();
    while ( iter != iend ) {
      const DTCCBId& ccbId( iter->first );
//      if ( ( ccbId.  wheelId ==   wheelId ) &&
//           ( ccbId.stationId == stationId ) &&
//           ( ccbId. sectorId ==  sectorId ) ) tempList.push_back( *iter );
      if ( ( ccbId.  wheelId !=   wheelId ) ||
           ( ccbId.stationId != stationId ) ||
           ( ccbId. sectorId !=  sectorId ) ) tempList.push_back( *iter );
      ++iter;
    }
    dataList = tempList;
    DTCCBId ccbId;
    ccbId.  wheelId =   wheelId;
    ccbId.stationId = stationId;
    ccbId. sectorId =  sectorId;
    std::vector<int>::const_iterator cfgIter = confKey.begin();
    std::vector<int>::const_iterator cfgIend = confKey.end();
    while ( cfgIter != cfgIend ) dataList.push_back( std::pair<DTCCBId,int>(
                                                     ccbId, *cfgIter++ ) );
    *confPtr = confKey;
    return -1;
  }
  else {
    dBuf->insert( chanKey.begin(),
                  chanKey.end(), new std::vector<int>( confKey ) );
    DTCCBId ccbId;
    ccbId.  wheelId =   wheelId;
    ccbId.stationId = stationId;
    ccbId. sectorId =  sectorId;
    std::vector<int>::const_iterator cfgIter = confKey.begin();
    std::vector<int>::const_iterator cfgIend = confKey.end();
    while ( cfgIter != cfgIend ) dataList.push_back( std::pair<DTCCBId,int>(
                                                     ccbId, *cfgIter++ ) );
    return 0;
  }

}


int DTCCBConfig::setConfigKey( const DTChamberId& id,
                               const std::vector<int>& confKey ) {
  return setConfigKey( id.wheel(),
                       id.station(),
                       id.sector(),
                       confKey );
}


int DTCCBConfig::appendConfigKey( int   wheelId,
                                  int stationId,
                                  int  sectorId,
                                  const std::vector<int>& confKey ) {

//  std::string mName = mapName();
//  DTBufferTree< int,std::vector<int>* >* dBuf =
//  DTDataBuffer< int,std::vector<int>* >::findBuffer( mName );
//  if ( dBuf == 0 ) {
//    cacheMap();
//    dBuf =
//    DTDataBuffer< int,std::vector<int>* >::findBuffer( mName );
//  }
  if ( dBuf == 0 ) cacheMap();

  std::vector<int> chanKey;
  chanKey.reserve(3);
  chanKey.push_back(   wheelId );
  chanKey.push_back( stationId );
  chanKey.push_back(  sectorId );

  DTCCBId ccbId;
  ccbId.  wheelId =   wheelId;
  ccbId.stationId = stationId;
  ccbId. sectorId =  sectorId;
  std::vector<int>::const_iterator iter = confKey.begin();
  std::vector<int>::const_iterator iend = confKey.end();
  int key;

  std::vector<int>* confPtr;
  int searchStatus = dBuf->find( chanKey.begin(), chanKey.end(), confPtr );

  if ( searchStatus ) {
    dBuf->insert( chanKey.begin(),
                  chanKey.end(), confPtr = new std::vector<int>( confKey ) );
  }

  while ( iter != iend ) {
    key = *iter++;
    dataList.push_back( std::pair<DTCCBId,int>( ccbId, key ) );
    confPtr->push_back( key );
  }

  if ( !searchStatus ) {
    return -1;
  }
  else {
    return 0;
  }

}


int DTCCBConfig::appendConfigKey( const DTChamberId& id,
                                  const std::vector<int>& confKey ) {
  return appendConfigKey( id.wheel(),
                          id.station(),
                          id.sector(),
                          confKey );
}


DTCCBConfig::const_iterator DTCCBConfig::begin() const {
  return dataList.begin();
}


DTCCBConfig::const_iterator DTCCBConfig::end() const {
  return dataList.end();
}


/*
std::string DTCCBConfig::mapName() const {
  std::stringstream name;
  name << dataVersion << "_map_CCBConfig" << this;
  return name.str();
}
*/


void DTCCBConfig::cacheMap() const {

//  std::string mName = mapName();
//  DTBufferTree< int,std::vector<int>* >* dBuf =
//  DTDataBuffer< int,std::vector<int>* >::openBuffer( mName );
  DTBufferTree< int,std::vector<int>* >** pBuf;
  pBuf = const_cast<DTBufferTree< int,std::vector<int>* >**>( &dBuf );
  *pBuf = new DTBufferTree< int,std::vector<int>* >;
  dBuf->setDefault( 0 );

  const_iterator iter = dataList.begin();
  const_iterator iend = dataList.end();
  std::vector<int> chanKey;
  chanKey.reserve(3);
  while ( iter != iend ) {

    const DTCCBId& chan = iter->first;

    chanKey.clear();
    chanKey.push_back( chan.  wheelId );
    chanKey.push_back( chan.stationId );
    chanKey.push_back( chan. sectorId );
    std::vector<int>* ccbConfPtr;
    int searchStatus = dBuf->find( chanKey.begin(),
                                   chanKey.end(), ccbConfPtr );
    if ( searchStatus ) dBuf->insert( chanKey.begin(),
                                      chanKey.end(),
                                      ccbConfPtr = new std::vector<int> );
    ccbConfPtr->push_back( iter->second );

    iter++;

  }

}


void DTCCBConfig::resetMap() const {
//  std::string mName = mapName();
//  DTBufferTree< int,std::vector<int>* >* dBuf =
//  DTDataBuffer< int,std::vector<int>* >::findBuffer( mName );
  if ( dBuf != 0 ) {
    std::vector< std::vector<int>* > list( dBuf->contList() );
    std::vector< std::vector<int>* >::const_iterator iter = list.begin();
    std::vector< std::vector<int>* >::const_iterator iend = list.end();
    while ( iter != iend ) delete *iter++;
  }
  return;
}


