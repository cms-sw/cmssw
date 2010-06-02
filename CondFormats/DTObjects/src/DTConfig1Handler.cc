/*
 *  See header file for a description of this class.
 *
 *  TEMPORARY TOOL TO HANDLE CONFIGURATIONS
 *  TO BE REMOVED IN FUTURE RELEASES
 *
 *
 *  $Date: 2009/06/05 09:31:22 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondFormats/DTObjects/interface/DTConfig1Handler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "CondCore/DBCommon/interface/TypedRef.h"

#include "CondFormats/DTObjects/interface/DTDB1Session.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <cstdio>

//-------------------
// Initializations --
//-------------------
int DTConfig1Handler::maxBrickNumber  = 50;
int DTConfig1Handler::maxStringNumber = 1000;
int DTConfig1Handler::maxByteNumber   = 1000000;
DTConfig1Handler::handler_map DTConfig1Handler::handlerMap;

//----------------
// Constructors --
//----------------
DTConfig1Handler::DTConfig1Handler( DTDB1Session* session,
                                  const std::string& token ):
  dbSession( session ),
  objToken( token ),
  refSet( 0 ),
  cachedBrickNumber(  0 ),
  cachedStringNumber( 0 ),
  cachedByteNumber(   0 ) {
}


//--------------
// Destructor --
//--------------
DTConfig1Handler::~DTConfig1Handler() {
  purge();
  delete refSet;
  refSet = 0;
}


//--------------
// Operations --
//--------------
/// create static object
DTConfig1Handler* DTConfig1Handler::create( DTDB1Session* session,
                                          const std::string& token ) {

  DTConfig1Handler* handlerPtr = 0;
//  c_map_iter iter = handlerMap.find( 
//                    reinterpret_cast<unsigned int>( session ) );
//  if ( iter == handlerMap.end() ) 
//       handlerMap.insert( std::pair<unsigned int,DTConfig1Handler*>(
//                          reinterpret_cast<unsigned int>( session ),
//                          handlerPtr = new DTConfig1Handler( session, 
//                                                            token ) ) );
  c_map_iter iter = handlerMap.find( session );
  if ( iter == handlerMap.end() ) 
       handlerMap.insert( std::pair<const DTDB1Session*,DTConfig1Handler*>(
                          session,
                          handlerPtr = new DTConfig1Handler( session, 
                                                            token ) ) );
  else                    handlerPtr = iter->second;
  return handlerPtr;

}


void DTConfig1Handler::remove( const DTDB1Session* session ) {
//  map_iter iter = handlerMap.find(
//                  reinterpret_cast<unsigned int>( session ) );
  map_iter iter = handlerMap.find( session );
  if ( iter != handlerMap.end() ) {
    delete iter->second;
    handlerMap.erase( iter );
  }
  return;
}


void DTConfig1Handler::remove( const DTConfig1Handler* handler ) {
  map_iter iter = handlerMap.begin();
  map_iter iend = handlerMap.end();
  while ( iter != iend ) {
    if ( iter->second == handler ) {
      handlerMap.erase( iter );
      delete handler;
      return;
    }
    iter++;
  }
  return;
}


/// get content
const DTConfigList* DTConfig1Handler::getContainer() {
  if ( refSet == 0 )
       refSet = new cond::TypedRef<DTConfigList>( *dbSession->poolDB(),
                                                  objToken );
  return refSet->ptr();
}


int DTConfig1Handler::get( int cfgId, DTConfigData*& obj ) {

  getContainer();
  const_iterator iter = refMap.find( cfgId );

  if ( iter != refMap.end() ) {
    obj = iter->second->ptr();
    return 0;
  }
  else {
    DTConfigToken token;
    if ( refSet->ptr()->get( cfgId, token ) == 0 ) {
	std::string actualToken( token.ref );
        int actualId = token.id;
        if ( ( actualToken == std::string( "dummyToken" ) ) &&
             ( actualId > 0 ) ) {
          refSet->ptr()->get( -999999999, token );
          actualToken = compToken( token.ref, actualId );
        }
        if ( ( cachedBrickNumber  > maxBrickNumber  ) ||
             ( cachedStringNumber > maxStringNumber ) ||
             ( cachedByteNumber   > maxByteNumber   ) ) purge(); //autoPurge();
        cond::TypedRef<DTConfigData>* refObj =
              new cond::TypedRef<DTConfigData>( *dbSession->poolDB(),
                                                actualToken );
        refMap.insert( std::pair<int,cond::TypedRef<DTConfigData>*>(
                       cfgId, refObj ) );
        obj = refObj->ptr();
        DTConfigData::data_iterator d_iter = obj->dataBegin();
        DTConfigData::data_iterator d_iend = obj->dataEnd();
        cachedBrickNumber++;
        cachedStringNumber += ( d_iend - d_iter );
        while ( d_iter != d_iend ) cachedByteNumber += ( *d_iter++ ).size();

        return -1;
    }
    else {
      obj = 0;
      return -2;
    }
  }
  return 999;

}


void DTConfig1Handler::getData( int cfgId,
                               std::vector<const std::string*>& list ) {
  DTConfigData* obj = 0;
  get( cfgId, obj );
  if ( obj == 0 ) return; 
  DTConfigData::data_iterator d_iter = obj->dataBegin();
  DTConfigData::data_iterator d_iend = obj->dataEnd();
  while ( d_iter != d_iend ) list.push_back( &( *d_iter++ ) );
  DTConfigData::link_iterator l_iter = obj->linkBegin();
  DTConfigData::link_iterator l_iend = obj->linkEnd();
  while ( l_iter != l_iend ) getData( *l_iter++, list );
  return;
}


int DTConfig1Handler::set( int cfgId, const std::string& token ) {
  getContainer();
  DTConfigToken configToken;
  configToken.id = compToken( token );
  configToken.ref = token;
  DTConfigList* rs = refSet->ptr();
  int status = rs->set( cfgId, configToken );
  refSet->markUpdate();
  return status;
}


DTConfig1Handler::const_iterator DTConfig1Handler::begin() const {
  return refMap.begin();
}


DTConfig1Handler::const_iterator DTConfig1Handler::end() const {
  return refMap.end();
}

void DTConfig1Handler::purge() {
  std::cout << "DTConfig1Handler::purge "
            << this << " " << dbSession << " "
            << cachedBrickNumber  << " "
            << cachedStringNumber << " "
            << cachedByteNumber   << std::endl;
  const_iterator iter = refMap.begin();
  const_iterator iend = refMap.end();
  while ( iter != iend ) {
    delete iter->second;
    *iter++;
  }
  refMap.clear();
  cachedBrickNumber  = 0;
  cachedStringNumber = 0;
  cachedByteNumber   = 0;
  return;
}


std::string DTConfig1Handler::clone( DTDB1Session* newSession,
                                    const std::string& objContainer,
                                    const std::string& refContainer ) {
  const DTConfigList* rs = getContainer();
  DTConfigList* rn = new DTConfigList( rs->version() );
  DTConfigList::const_iterator iter = rs->begin();
  DTConfigList::const_iterator iend = rs->end();
  while ( iter != iend ) {
    const DTConfigToken& token = iter->second;
    std::string actualToken( token.ref );
    int actualId = token.id;
    if ( ( actualToken == std::string( "dummyToken" ) ) &&
         ( actualId > 0 ) ) {
      DTConfigToken chkToken;
      refSet->ptr()->get( -999999999, chkToken );
      actualToken = compToken( chkToken.ref, actualId );
    }
    cond::TypedRef<DTConfigData> sRef( *dbSession->poolDB(),
                                       actualToken );
    std::cout << " copying brick " << sRef->getId() << std::endl;
    DTConfigData* ptr = new DTConfigData( *( sRef.ptr() ) );
    cond::TypedRef<DTConfigData> dRef( *newSession->poolDB(), ptr );
    dRef.markWrite( refContainer );
    DTConfigToken objToken;
    std::string tokenRef( dRef.token() );
    objToken.ref = tokenRef;
    objToken.id = compToken( tokenRef );
    rn->set( iter->first, objToken );
    iter++;
  }
  cond::TypedRef<DTConfigList> setRef( *newSession->poolDB(), rn );
  setRef.markWrite( objContainer );
  std::string token = setRef.token();
  return token;
}


std::string DTConfig1Handler::clone( DTDB1Session* newSession,
                                    const std::string& newToken,
                                    const std::string& objContainer,
                                    const std::string& refContainer ) {

  if ( !newToken.length() ) return clone( newSession,
                                          objContainer, refContainer );

  const DTConfigList* rs = getContainer();
  std::cout << "look for existing list reference..." << std::endl;
  cond::TypedRef<DTConfigList>* setRef = new cond::TypedRef<DTConfigList>(
                                         *newSession->poolDB(), newToken );
  std::cout << "look for existing list pointer..." << std::endl;
  DTConfigList* rn = 0;
    rn = setRef->ptr();
    std::cout << "existing list pointer got " << rn << std::endl;
  bool containerMissing = ( rn == 0 );
  if ( containerMissing ) rn = new DTConfigList( rs->version() );

  DTConfigList::const_iterator iter = rs->begin();
  DTConfigList::const_iterator iend = rs->end();
  while ( iter != iend ) {
    const DTConfigToken& token = iter->second;
    std::string actualToken( token.ref );
    int actualId = token.id;
    if ( ( actualToken == std::string( "dummyToken" ) ) &&
         ( actualId > 0 ) ) {
      DTConfigToken chkToken;
      refSet->ptr()->get( -999999999, chkToken );
      actualToken = compToken( chkToken.ref, actualId );
    }
    cond::TypedRef<DTConfigData> sRef( *dbSession->poolDB(),
                                       actualToken );
    int cfgId = sRef->getId();
    std::cout << " checking brick " << cfgId << std::endl;
    DTConfigToken dumToken;
    if ( rn->get( cfgId, dumToken ) ) {
      std::cout << " ... copying brick " << cfgId << std::endl;
      DTConfigData* ptr = new DTConfigData( *( sRef.ptr() ) );
      cond::TypedRef<DTConfigData> dRef( *newSession->poolDB(), ptr );
      dRef.markWrite( refContainer );
      DTConfigToken objToken;
      std::string tokenRef( dRef.token() );
      objToken.ref = tokenRef;
      objToken.id = compToken( tokenRef );
      rn->set( iter->first, objToken );
    }
    iter++;
  }
  std::string token( "" );
  if ( containerMissing ) {
    setRef = new cond::TypedRef<DTConfigList>( *newSession->poolDB(), rn );
    setRef->markWrite( objContainer );
    token = setRef->token();
  }
  else {
    setRef->markUpdate();
  }
  return token;
}


std::string DTConfig1Handler::compToken( const std::string& refToken, int id ) {
  std::string actualToken( "" );
  char* buf = new char[9];
  unsigned int iofb = 0;
  unsigned int iofe = 0;
  while ( true ) {
    iofb = refToken.find( "[", iofb );
//    if ( iofb == std::string::npos ) break;
    if ( iofb >= refToken.length() ) break;
    iofe = refToken.find( "]", iofb );
    std::string sub( refToken.substr( iofb, 1 + iofe - iofb ) );
    iofb = sub.find( "=", 0 );
    std::string key( sub.substr( 1, iofb ) );
    if ( key == std::string( "OID=" ) ) {
      iofb = 1 + sub.find( "-", 0 );
      std::string val( sub.substr( iofb, sub.length() - iofb - 1 ) );
      actualToken += sub.substr( 0, iofb );
      sprintf( buf, "%8.8x", id );
      actualToken += buf;
      actualToken += "]";
    }
    else {
      actualToken += sub;
    }
    iofb = iofe;
  }
  delete[] buf;
  return actualToken;
}


int DTConfig1Handler::compToken( const std::string& refToken ) {
  unsigned int iofb = 0;
  unsigned int iofe = 0;
  unsigned tokenId;
  while ( true ) {
    iofb = refToken.find( "[", iofb );
//    if ( iofb == std::string::npos ) break;
    if ( iofb >= refToken.length() ) break;
    iofe = refToken.find( "]", iofb );
    std::string sub( refToken.substr( iofb, 1 + iofe - iofb ) );
    iofb = sub.find( "=", 0 );
    std::string key( sub.substr( 1, iofb ) );
    if ( key == std::string( "OID=" ) ) {
      iofb = 1 + sub.find( "-", 0 );
      std::string val( sub.substr( iofb, sub.length() - iofb - 1 ) );
      sscanf( val.c_str(), "%x", &tokenId );
      break;
    }
    iofb = iofe;
  }
  return tokenId;
}


