/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTUserKeyedConfigHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/KeyedElement.h"
#include "CondCore/CondDB/interface/KeyList.h"

#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <memory>

//-------------------
// Initializations --
//-------------------
cond::persistency::KeyList* DTUserKeyedConfigHandler::keyList = nullptr;

//----------------
// Constructors --
//----------------
DTUserKeyedConfigHandler::DTUserKeyedConfigHandler( const edm::ParameterSet& ps ):
 dataRun(               ps.getParameter<int>         ( "run" ) ),
 dataTag(               ps.getParameter<std::string> ( "tag" ) ),
 onlineConnect(         ps.getParameter<std::string> ( "onlineDB" ) ),
 onlineAuthentication(  ps.getParameter<std::string> ( 
                        "onlineAuthentication" ) ),
 onlineAuthSys( ps.getUntrackedParameter<int>( "onlineAuthSys",1 ) ),
 brickContainer(        ps.getParameter<std::string> ( "container" ) ),
 writeKeys(             ps.getParameter<bool>        ( "writeKeys" ) ),
 writeData(             ps.getParameter<bool>        ( "writeData" ) ),
 connection(),
 isession() {
  std::cout << " PopCon application for DT configuration export "
            <<  onlineAuthentication << std::endl;

  std::vector<edm::ParameterSet> dtConfigKeys (
        ps.getParameter< std::vector<edm::ParameterSet> >( "DTConfigKeys" )
  );
  std::vector<edm::ParameterSet>::const_iterator iter = dtConfigKeys.begin();
  std::vector<edm::ParameterSet>::const_iterator iend = dtConfigKeys.end();
  while ( iter != iend ) {
    const edm::ParameterSet& cp = *iter++;
    int configType = cp.getUntrackedParameter<int>( "configType" );
    int configKey  = cp.getUntrackedParameter<int>( "configKey"  );
    std::cout << "config: "
              << configType << " -> "
              << configKey << std::endl;
    DTConfigKey userKey;
    userKey.confType = configType;
    userKey.confKey  = configKey;
    userConf.push_back( userKey );
  }
}

//--------------
// Destructor --
//--------------
DTUserKeyedConfigHandler::~DTUserKeyedConfigHandler() {
}

//--------------
// Operations --
//--------------
void DTUserKeyedConfigHandler::getNewObjects() {

  //to access the information on the tag inside the offline database:
  cond::TagInfo const & ti = tagInfo();
  unsigned int last = ti.lastInterval.first;
  std::cout << "last configuration key already copied for run: "
            << last << std::endl;

  std::vector<DTConfigKey> lastKey;
//  std::cout << "check for last " << std::endl;
  if ( last == 0 ) {
    DTCCBConfig* dummyConf = new DTCCBConfig( dataTag );
    dummyConf->setStamp( 0 );
    dummyConf->setFullKey( lastKey );
    cond::Time_t snc = 1;
    if ( writeKeys && ( dataRun > 1 ) )
         m_to_transfer.push_back( std::make_pair( dummyConf, snc ) );
  }
  else {
    std::cout << "get last payload" << std::endl;
    Ref payload = lastPayload();
    std::cout << "get last full key" << std::endl;
    lastKey = payload->fullKey();
    std::cout << "last key: " << std::endl;
    std::vector<DTConfigKey>::const_iterator keyIter = lastKey.begin();
    std::vector<DTConfigKey>::const_iterator keyIend = lastKey.end();
    while ( keyIter != keyIend ) {
      const DTConfigKey& keyList = *keyIter++;
      std::cout << keyList.confType << " : " << keyList.confKey << std::endl;
    }
  }

  std::cout << "configure DbConnection" << std::endl;
  //  conn->configure( cond::CmsDefaults );
  connection.setAuthenticationPath( onlineAuthentication );
  connection.setAuthenticationSystem( onlineAuthSys );
  connection.configure();
  std::cout << "create/open DbSession" << std::endl;
  isession = connection.createCoralSession( onlineConnect  );
  std::cout << "start transaction" << std::endl;
  isession->transaction().start( true );
  
  // get ccb identifiers map
  std::cout << "retrieve CCB map" << std::endl;
  std::map<int,DTCCBId> ccbMap;
  coral::ITable& ccbMapTable =
    isession->nominalSchema().tableHandle( "CCBMAP" );
  std::unique_ptr<coral::IQuery>
    ccbMapQuery( ccbMapTable.newQuery() );
  ccbMapQuery->addToOutputList( "CCBID" );
  ccbMapQuery->addToOutputList( "WHEEL" );
  ccbMapQuery->addToOutputList( "SECTOR" );
  ccbMapQuery->addToOutputList( "STATION" );
  coral::ICursor& ccbMapCursor = ccbMapQuery->execute();
  while( ccbMapCursor.next() ) {
    const coral::AttributeList& row = ccbMapCursor.currentRow();
    int ccb     = row["CCBID"  ].data<int>();
    int wheel   = row["WHEEL"  ].data<int>();
    int sector  = row["SECTOR" ].data<int>();
    int station = row["STATION"].data<int>();
    DTCCBId ccbId;
    ccbId.  wheelId =   wheel;
    ccbId.stationId = station;
    ccbId. sectorId =  sector;
    ccbMap.insert( std::pair<int,DTCCBId>( ccb, ccbId ) );
  }

  // get brick types
  std::cout << "retrieve brick types" << std::endl;
  std::map<int,int> bktMap;
  coral::AttributeList emptyBindVariableList;
  std::unique_ptr<coral::IQuery>
         brickTypeQuery( isession->nominalSchema().newQuery() );
  brickTypeQuery->addToTableList( "CFGBRICKS" );
  brickTypeQuery->addToTableList( "BRKT2CSETT" );
  std::string bTypeCondition = "CFGBRICKS.BRKTYPE=BRKT2CSETT.BRKTYPE";
  brickTypeQuery->addToOutputList( "CFGBRICKS.BRKID" );
  brickTypeQuery->addToOutputList( "BRKT2CSETT.CSETTYPE" );
  brickTypeQuery->setCondition( bTypeCondition, emptyBindVariableList );
  coral::ICursor& brickTypeCursor = brickTypeQuery->execute();
  while( brickTypeCursor.next() ) {
    const coral::AttributeList& row = brickTypeCursor.currentRow();
    int id = row["CFGBRICKS.BRKID"    ].data<int>();
    int bt = row["BRKT2CSETT.CSETTYPE"].data<short>();
//    std::cout << "brick " << id << " type " << bt << std::endl;
// @@FIX - TEMPORARY PATCH
//    if ( bt > 3 ) bt = 3;
    bktMap.insert( std::pair<int,int>( id, bt ) );
  }

  // get ccb config keys
  std::cout << "retrieve CCB configuration keys" << std::endl;
  std::map<int,std::map<int,int>*> keyMap;
  std::map<int,int> cckMap;
  coral::ITable& ccbRelTable =
    isession->nominalSchema().tableHandle( "CCBRELATIONS" );
  std::unique_ptr<coral::IQuery>
    ccbRelQuery( ccbRelTable.newQuery() );
  ccbRelQuery->addToOutputList( "CONFKEY" );
  ccbRelQuery->addToOutputList( "CCBID" );
  ccbRelQuery->addToOutputList( "CONFCCBKEY" );
  coral::ICursor& ccbRelCursor = ccbRelQuery->execute();
  // loop over all full configurations
  while( ccbRelCursor.next() ) {
    const coral::AttributeList& row = ccbRelCursor.currentRow();
    int cfg     = row["CONFKEY"   ].data<int>();
    int ccb     = row["CCBID"     ].data<int>();
    int key     = row["CONFCCBKEY"].data<int>();
    // check for used configurations
//    if ( cfgMap.find( cfg ) == cfgMap.end() ) continue;
    if ( userDiscardedKey( cfg ) ) continue;
    std::map<int,std::map<int,int>*>::const_iterator keyIter =
                                                     keyMap.find( cfg );
    std::map<int,std::map<int,int>*>::const_iterator keyIend =
                                                     keyMap.end();
    std::map<int,int>* mapPtr = nullptr;
    // check for new full configuration
    if ( keyIter != keyIend ) mapPtr = keyIter->second;
    else                      keyMap.insert(
                              std::pair<int,std::map<int,int>*>( cfg,
                              mapPtr = new std::map<int,int> ) );
    // store ccb config key
    std::map<int,int>& mapRef( *mapPtr );
    mapRef.insert( std::pair<int,int>( ccb, key ) );
    // check for new ccb config key
    if ( cckMap.find( key ) == cckMap.end() )
         cckMap.insert( std::pair<int,int>( key, ccb ) );
  }

  // get brick keys
  std::cout << "retrieve CCB configuration bricks" << std::endl;
  std::map<int,std::vector<int>*> brkMap;
  coral::ITable& confBrickTable =
    isession->nominalSchema().tableHandle( "CFG2BRKREL" );
  std::unique_ptr<coral::IQuery>
    confBrickQuery( confBrickTable.newQuery() );
  confBrickQuery->addToOutputList( "CONFID" );
  confBrickQuery->addToOutputList( "BRKID"  );
  coral::ICursor& confBrickCursor = confBrickQuery->execute();
  // loop over all brick keys
  while( confBrickCursor.next() ) {
    const coral::AttributeList& row = confBrickCursor.currentRow();
    int key = row["CONFID"].data<int>();
    int brk = row["BRKID" ].data<int>();
    // check for used ccb config key
    if ( cckMap.find( key ) == cckMap.end() ) continue;
    std::map<int,std::vector<int>*>::const_iterator brkIter =
                                                    brkMap.find( key );
    std::map<int,std::vector<int>*>::const_iterator brkIend =
                                                    brkMap.end();
    // check for new ccb config key
    std::vector<int>* brkPtr = nullptr;
    if ( brkIter != brkIend ) brkPtr = brkIter->second;
    else                      brkMap.insert(
                              std::pair<int,std::vector<int>*>( key,
                              brkPtr = new std::vector<int> ) );
    // store brick key
    brkPtr->push_back( brk );
  }

  // set run and full configuration in payload
  DTCCBConfig* fullConf = new DTCCBConfig( dataTag );
  fullConf->setStamp( 1 );
  fullConf->setFullKey( userConf );
  std::map<int,bool> userBricks;
  std::map<int,bool>::const_iterator uBrkIter = userBricks.begin();
  std::map<int,bool>::const_iterator uBrkIend = userBricks.end();
  std::vector<DTConfigKey>::const_iterator cfgIter = userConf.begin();
  std::vector<DTConfigKey>::const_iterator cfgIend = userConf.end();
  while ( cfgIter != cfgIend ) {
    const DTConfigKey& cfgEntry = *cfgIter++;
    int cft = cfgEntry.confType;
    int cfg = cfgEntry.confKey;
    // retrieve ccb config map
    std::map<int,std::map<int,int>*>::const_iterator keyIter =
                                                     keyMap.find( cfg );
    std::map<int,std::map<int,int>*>::const_iterator keyIend =
                                                     keyMap.end();
    std::map<int,int>* mapPtr = nullptr;
    if ( keyIter != keyIend ) mapPtr = keyIter->second;
    if ( mapPtr == nullptr ) continue;
    std::map<int,int>::const_iterator ccbIter = mapPtr->begin();
    std::map<int,int>::const_iterator ccbIend = mapPtr->end();
    while ( ccbIter != ccbIend ) {
      const std::pair<int,int>& ccbEntry = *ccbIter++;
      // get ccb config key
      int ccb = ccbEntry.first;
      int key = ccbEntry.second;
      // retrieve chamber id
      std::map<int,DTCCBId>::const_iterator ccbIter = ccbMap.find( ccb );
      std::map<int,DTCCBId>::const_iterator ccbIend = ccbMap.end();
      if ( ccbIter == ccbIend ) continue;
      const DTCCBId& chaId = ccbIter->second;
      // retrieve brick id list
      std::map<int,std::vector<int>*>::const_iterator brkIter =
                                                      brkMap.find( key );
      std::map<int,std::vector<int>*>::const_iterator brkIend =
                                                      brkMap.end();
      if ( brkIter == brkIend ) continue;
      std::vector<int>* brkPtr = brkIter->second;
      if ( brkPtr == nullptr ) continue;
      // brick id lists in payload
      std::vector<int> bkList;
      bkList.reserve( 20 );
      std::map<int,int>::const_iterator bktIter = bktMap.begin();
      std::map<int,int>::const_iterator bktIend = bktMap.end();
      std::vector<int>::const_iterator bkiIter = brkPtr->begin();
      std::vector<int>::const_iterator bkiIend = brkPtr->end();
      while ( bkiIter != bkiIend ) {
        int brickId = *bkiIter++;
        bktIter = bktMap.find( brickId );
	if ( bktIter == bktIend ) continue;
        if ( bktIter->second == cft ) {
          bkList.push_back( brickId );
          uBrkIter = userBricks.find( brickId );
          if ( uBrkIter == uBrkIend ) 
              userBricks.insert( std::pair<int,bool>( brickId, true ) );
        }
      }
      fullConf->appendConfigKey( chaId.wheelId,
                                 chaId.stationId,
                                 chaId.sectorId,
                                 bkList );
    }
  }
  cond::Time_t snc = dataRun;
  if ( writeKeys ) m_to_transfer.push_back( std::make_pair( fullConf, snc ) );
  std::cout << "writing payload : " << sizeof( *fullConf ) 
            << " ( " << ( fullConf->end() - fullConf->begin() )
            << " ) " << std::endl;
  if ( writeData ) chkConfigList( userBricks );

  isession->transaction().commit();

  return;

}


void DTUserKeyedConfigHandler::chkConfigList(
                               const std::map<int,bool>& userBricks ) {

  std::cout << "open POOL out db " << std::endl;
  edm::Service<cond::service::PoolDBOutputService> outdb;

  std::map<int,bool>::const_iterator uBrkIter = userBricks.begin();
  std::map<int,bool>::const_iterator uBrkIend = userBricks.end();

  coral::ITable& brickConfigTable =
    isession->nominalSchema().tableHandle( "CFGBRICKS" );
  std::unique_ptr<coral::IQuery>
    brickConfigQuery( brickConfigTable.newQuery() );
  brickConfigQuery->addToOutputList( "BRKID" );
  brickConfigQuery->addToOutputList( "BRKNAME" );
  coral::ICursor& brickConfigCursor = brickConfigQuery->execute();
  DTKeyedConfig* brickData = nullptr;
  std::vector<int> missingList;
  std::vector<unsigned long long> checkedKeys;
  while( brickConfigCursor.next() ) {
    const coral::AttributeList& row = brickConfigCursor.currentRow();
    int brickConfigId = row["BRKID"].data<int>();
    uBrkIter = userBricks.find( brickConfigId );
    if ( uBrkIter == uBrkIend ) continue;
    if ( !( uBrkIter->second ) ) continue;
    std::string brickConfigName = row["BRKNAME"].data<std::string>();
    std::cout << "brick " << brickConfigId
              << " : "    << brickConfigName << std::endl;
    checkedKeys.push_back( brickConfigId );
    bool brickFound = false;
    try {
      std::cout << "load brick " << checkedKeys[0] << std::endl;
      std::cout << "key list " <<  keyList << std::endl;
      keyList->load( checkedKeys );
      std::cout << "get brick..." << std::endl;
      std::shared_ptr<DTKeyedConfig> brickCheck =
                           keyList->get<DTKeyedConfig>( 0 );
      if ( brickCheck.get() ) {
	brickFound = ( brickCheck->getId() == brickConfigId );
      }
    }
    catch ( std::exception const& ) {
    }
    if ( !brickFound ) {
      std::cout << "brick " << brickConfigId << " missing, copy request"
                << std::endl;
      missingList.push_back( brickConfigId );
    }
    checkedKeys.clear();
  }
  keyList->load( checkedKeys );

  std::vector<int>::const_iterator brickIter = missingList.begin();
  std::vector<int>::const_iterator brickIend = missingList.end();
  while ( brickIter != brickIend ) {
    int brickConfigId = *brickIter++;
    coral::AttributeList bindVariableList;
    bindVariableList.extend( "brickId", typeid(int) );
    bindVariableList["brickId"].data<int>() = brickConfigId;
    std::unique_ptr<coral::IQuery>
           brickDataQuery( isession->nominalSchema().newQuery() );
    brickDataQuery->addToTableList( "CFGRELATIONS" );
    brickDataQuery->addToTableList( "CONFIGCMDS" );
    std::string
    brickCondition  =      "CONFIGCMDS.CMDID=CFGRELATIONS.CMDID";
    brickCondition += " and CFGRELATIONS.BRKID=:brickId";
    brickDataQuery->addToOutputList( "CFGRELATIONS.BRKID" );
    brickDataQuery->addToOutputList( "CONFIGCMDS.CONFDATA" );
    brickDataQuery->setCondition( brickCondition, bindVariableList );
    coral::ICursor& brickDataCursor = brickDataQuery->execute();
    brickData = new DTKeyedConfig();
    brickData->setId( brickConfigId );
    while( brickDataCursor.next() ) {
      const coral::AttributeList& row = brickDataCursor.currentRow();
      brickData->add( row["CONFIGCMDS.CONFDATA"].data<std::string>() );
    }
    cond::KeyedElement k( brickData, brickConfigId );
    std::cout << "now writing brick: " << brickConfigId << std::endl;
    outdb->writeOne( k.m_obj, k.m_key, brickContainer );
  }

  return;

}


std::string DTUserKeyedConfigHandler::id() const {
  return dataTag;
}


bool DTUserKeyedConfigHandler::sameConfigList(
                         const std::vector<DTConfigKey>& cfgl,
                         const std::vector<DTConfigKey>& cfgr ) {
  if ( cfgl.size() != cfgr.size() ) return false;
  std::map<int,int> lmap;
  std::vector<DTConfigKey>::const_iterator lIter = cfgl.begin();
  std::vector<DTConfigKey>::const_iterator lIend = cfgl.end();
  while ( lIter != lIend ) {
    const DTConfigKey& entry = *lIter++;
    lmap.insert( std::pair<int,int>( entry.confType, entry.confKey ) );
  }
  std::map<int,int> rmap;
  std::vector<DTConfigKey>::const_iterator rIter = cfgr.begin();
  std::vector<DTConfigKey>::const_iterator rIend = cfgr.end();
  while ( rIter != rIend ) {
    const DTConfigKey& entry = *rIter++;
    rmap.insert( std::pair<int,int>( entry.confType, entry.confKey ) );
  }
  std::map<int,int>::const_iterator lmIter = lmap.begin();
  std::map<int,int>::const_iterator lmIend = lmap.end();
  std::map<int,int>::const_iterator rmIter = rmap.begin();
  std::map<int,int>::const_iterator rmIend = rmap.end();
  while ( ( lmIter != lmIend ) &&
          ( rmIter != rmIend ) ) {
    const std::pair<int,int>& lEntry = *lmIter++;
    const std::pair<int,int>& rEntry = *rmIter++;
    if ( lEntry.first  != rEntry.first  ) return false;
    if ( lEntry.second != rEntry.second ) return false;
  }
  return true;
}


bool DTUserKeyedConfigHandler::userDiscardedKey( int key ) {
  std::vector<DTConfigKey>::const_iterator iter = userConf.begin();
  std::vector<DTConfigKey>::const_iterator iend = userConf.end();
  while ( iter != iend ) {
    const DTConfigKey& entry = *iter++;
    if ( entry.confKey == key ) return false;
  }
  return true;
}

void DTUserKeyedConfigHandler::setList( cond::persistency::KeyList* list ) {
  keyList = list;
}

