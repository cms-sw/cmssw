/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTKeyedConfigHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/KeyedElement.h"
#include "CondCore/CondDB/interface/KeyList.h"

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

//-------------------
// Initializations --
//-------------------
cond::persistency::KeyList* DTKeyedConfigHandler::keyList = 0;

//----------------
// Constructors --
//----------------
DTKeyedConfigHandler::DTKeyedConfigHandler( const edm::ParameterSet& ps ):
 copyData(   ps.getUntrackedParameter<bool> ( "copyData", true ) ),
 minBrickId( ps.getUntrackedParameter<int> ( "minBrick", 0 ) ),
 maxBrickId( ps.getUntrackedParameter<int> ( "maxBrick", 999999999 ) ),
 minRunId(   ps.getUntrackedParameter<int> ( "minRun", 0 ) ),
 maxRunId(   ps.getUntrackedParameter<int> ( "maxRun", 999999999 ) ),
 dataTag(               ps.getParameter<std::string> ( "tag" ) ),
 onlineConnect(         ps.getParameter<std::string> ( "onlineDB" ) ),
 onlineAuthentication(  ps.getParameter<std::string> ( 
                        "onlineAuthentication" ) ),
 brickContainer(        ps.getParameter<std::string> ( "container" ) ),
 connection(),
 isession() {
  std::cout << " PopCon application for DT configuration export "
            <<  onlineAuthentication << std::endl;
}

//--------------
// Destructor --
//--------------
DTKeyedConfigHandler::~DTKeyedConfigHandler() {
}

//--------------
// Operations --
//--------------
void DTKeyedConfigHandler::getNewObjects() {

  //to access the information on the tag inside the offline database:
  cond::TagInfo const & ti = tagInfo();
  unsigned int last = ti.lastInterval.first;
  std::cout << "last configuration key already copied for run: "
            << last << std::endl;

  std::vector<DTConfigKey> lastKey;
  std::cout << "check for last " << std::endl;
  if ( last == 0 ) {
    DTCCBConfig* dummyConf = new DTCCBConfig( dataTag );
    dummyConf->setStamp( 0 );
    dummyConf->setFullKey( lastKey );
    cond::Time_t snc = 1;
    std::cout << "write dummy " << std::endl;
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
      const DTConfigKey& cfgKeyList = *keyIter++;
      std::cout << cfgKeyList.confType << " : "
                << cfgKeyList.confKey << std::endl;
    }
  }

  //to access the information on last successful log entry for this tag:
//  cond::LogDBEntry const & lde = logDBEntry();     

  //to access the lastest payload (Ref is a smart pointer)
//  Ref payload = lastPayload();

  if ( !copyData ) return;

  unsigned lastRun = last;
  std::cout << "check for new runs since " << lastRun << std::endl;

  std::cout << "configure DbConnection" << std::endl;
  //  conn->configure( cond::CmsDefaults );
  connection.setAuthenticationPath( onlineAuthentication );
  connection.configure();
  std::cout << "create/open DbSession" << std::endl;
  isession = connection.createSession( onlineConnect );
  std::cout << "start transaction" << std::endl;
  isession.transaction().start();

  // =========== Update configuration data
  chkConfigList();

  // =========== Find latest runs
  std::cout << "get run config..." << std::endl;
  std::map<int,std::vector<int>*> runMap;
  std::map<int,std::vector<DTConfigKey>*> rhcMap;
  coral::ITable& runHistoryTable =
    isession.nominalSchema().tableHandle( "RUNHISTORY" );
  std::auto_ptr<coral::IQuery>
    runHistoryQuery( runHistoryTable.newQuery() );
  runHistoryQuery->addToOutputList( "RUN" );
  runHistoryQuery->addToOutputList( "RHID" );
  coral::ICursor& runHistoryCursor = runHistoryQuery->execute();
  while( runHistoryCursor.next() ) {
    const coral::AttributeList& row = runHistoryCursor.currentRow();
    int runId = row[    "RUN"].data<int>();
    int rhcId = static_cast<int>( row["RHID"].data<int>() );
    // ----------- ignore already copied runs
    if ( static_cast<unsigned>( runId ) <= lastRun ) continue;
    // ----------- ignore runs outside required range
    if ( runId < minRunId ) continue;
    if ( runId > maxRunId ) continue;
    std::cout << "schedule config key copy for run "
              << runId << " ---> RHID " << rhcId << std::endl;
    // ----------- retrieve or create RH relation list for this run
    std::vector<int>* rhlPtr = 0;
    std::map<int,std::vector<int>*>::const_iterator runIter;
    if ( ( runIter = runMap.find( runId ) ) != runMap.end() )
         rhlPtr = runIter->second;
    else runMap.insert( std::pair<int,std::vector<int>*>( runId,
         rhlPtr = new std::vector<int> ) );
    // ----------- append RH relation to the list
    rhlPtr->push_back( rhcId );
    // ----------- create key list for this RH relation
    if ( rhcMap.find( rhcId ) == rhcMap.end() )
         rhcMap.insert( std::pair<int,std::vector<DTConfigKey>*>( rhcId,
                        new std::vector<DTConfigKey> ) );
  }
  if ( !runMap.size() ) std::cout << "no new run found" << std::endl;

  // =========== get ccb identifiers map
  std::cout << "retrieve CCB map" << std::endl;
  std::map<int,DTCCBId> ccbMap;
  coral::ITable& ccbMapTable =
    isession.nominalSchema().tableHandle( "CCBMAP" );
  std::auto_ptr<coral::IQuery>
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

  // =========== get brick types
  std::cout << "retrieve brick types" << std::endl;
  std::map<int,int> bktMap;
  coral::AttributeList emptyBindVariableList;
  std::auto_ptr<coral::IQuery>
         brickTypeQuery( isession.nominalSchema().newQuery() );
  brickTypeQuery->addToTableList( "CFGBRICKS" );
  brickTypeQuery->addToTableList( "BRKT2CSETT" );
  // ----------- join brick type (1-11) to subsystem part (1-6)
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

  // =========== get RH relations
  std::cout << "retrieve RH relations" << std::endl;
  std::map<int,int> cfgMap;
  coral::ITable& rhcRelTable =
    isession.nominalSchema().tableHandle( "RHRELATIONS" );
  std::auto_ptr<coral::IQuery>
    rhcRelQuery( rhcRelTable.newQuery() );
  rhcRelQuery->addToOutputList( "RHID" );
  rhcRelQuery->addToOutputList( "CONFKEY" );
  rhcRelQuery->addToOutputList( "CSETTYPEID" );
  coral::ICursor& rhcRelCursor = rhcRelQuery->execute();
  while( rhcRelCursor.next() ) {
    const coral::AttributeList& row = rhcRelCursor.currentRow();
    int rhc     = row["RHID"      ].data<int>();
    int key     = row["CONFKEY"   ].data<int>();
    int cfg     = row["CSETTYPEID"].data<int>();
    // ----------- check for used RH relations, skip unused
    std::map<int,std::vector<DTConfigKey>*>::iterator rhcIter =
                                                      rhcMap.find( rhc );
    std::map<int,std::vector<DTConfigKey>*>::iterator rhcIend =
                                                      rhcMap.end();
    if ( rhcIter == rhcIend ) continue;
    std::vector<DTConfigKey>* keyPtr = rhcIter->second;
    DTConfigKey confList;
    confList.confType = cfg;
    confList.confKey  = key;
    keyPtr->push_back( confList );
    // ----------- flag config key as used
    if ( cfgMap.find( cfg ) == cfgMap.end() )
         cfgMap.insert( std::pair<int,int>( key, rhc ) );
  }

  // =========== get ccb config keys
  std::cout << "retrieve CCB configuration keys" << std::endl;
  std::map<int,std::map<int,int>*> keyMap;
  std::map<int,int> cckMap;
  coral::ITable& ccbRelTable =
    isession.nominalSchema().tableHandle( "CCBRELATIONS" );
  std::auto_ptr<coral::IQuery>
    ccbRelQuery( ccbRelTable.newQuery() );
  ccbRelQuery->addToOutputList( "CONFKEY" );
  ccbRelQuery->addToOutputList( "CCBID" );
  ccbRelQuery->addToOutputList( "CONFCCBKEY" );
  coral::ICursor& ccbRelCursor = ccbRelQuery->execute();
  while( ccbRelCursor.next() ) {
    const coral::AttributeList& row = ccbRelCursor.currentRow();
    int cfg     = row["CONFKEY"   ].data<int>();
    int ccb     = row["CCBID"     ].data<int>();
    int key     = row["CONFCCBKEY"].data<int>();
    // ----------- check for used configurations, skip unused
    if ( cfgMap.find( cfg ) == cfgMap.end() ) continue;
    // ----------- retrieve or create ccb id-key map for this configuration
    std::map<int,int>* mapPtr = 0;
    std::map<int,std::map<int,int>*>::const_iterator keyIter;
    if ( ( keyIter = keyMap.find( cfg ) ) != keyMap.end() )
         mapPtr = keyIter->second;
    else keyMap.insert( std::pair<int,std::map<int,int>*>( cfg,
         mapPtr = new std::map<int,int> ) );
//    std::map<int,std::map<int,int>*>::const_iterator keyIter =
//                                                     keyMap.find( cfg );
//    std::map<int,std::map<int,int>*>::const_iterator keyIend =
//                                                     keyMap.end();
//    if ( keyIter != keyIend ) mapPtr = keyIter->second;
//    else                      keyMap.insert(
//                              std::pair<int,std::map<int,int>*>( cfg,
//                              mapPtr = new std::map<int,int> ) );
    // ----------- store config key for this ccb
    std::map<int,int>& mapRef( *mapPtr );
    mapRef.insert( std::pair<int,int>( ccb, key ) );
    // ----------- flag ccb config key as used
    if ( cckMap.find( key ) == cckMap.end() )
         cckMap.insert( std::pair<int,int>( key, ccb ) );
  }

  // =========== get brick keys
  std::cout << "retrieve CCB configuration bricks" << std::endl;
  std::map<int,std::vector<int>*> brkMap;
  coral::ITable& confBrickTable =
    isession.nominalSchema().tableHandle( "CFG2BRKREL" );
  std::auto_ptr<coral::IQuery>
    confBrickQuery( confBrickTable.newQuery() );
  confBrickQuery->addToOutputList( "CONFID" );
  confBrickQuery->addToOutputList( "BRKID"  );
  coral::ICursor& confBrickCursor = confBrickQuery->execute();
  while( confBrickCursor.next() ) {
    const coral::AttributeList& row = confBrickCursor.currentRow();
    int key = row["CONFID"].data<int>();
    int brk = row["BRKID" ].data<int>();
    // ----------- check for used ccb config key, skip unused
    if ( cckMap.find( key ) == cckMap.end() ) continue;
//    std::map<int,std::vector<int>*>::const_iterator brkIter =
//                                                    brkMap.find( key );
//    std::map<int,std::vector<int>*>::const_iterator brkIend =
//                                                    brkMap.end();
    // ----------- retrieve or create brick list for this ccb config
    std::vector<int>* brkPtr = 0;
    std::map<int,std::vector<int>*>::const_iterator brkIter;
//check for new ccb config key
    if ( ( brkIter = brkMap.find( key ) ) != brkMap.end() )
         brkPtr = brkIter->second;
    else brkMap.insert( std::pair<int,std::vector<int>*>( key,
         brkPtr = new std::vector<int> ) );
//    if ( brkIter != brkIend ) brkPtr = brkIter->second;
//    else                      brkMap.insert(
//                              std::pair<int,std::vector<int>*>( key,
//                              brkPtr = new std::vector<int> ) );
    // ----------- store brick key
    brkPtr->push_back( brk );
  }

  // =========== loop over new runs
  std::map<int,std::vector<int>*>::const_iterator runIter = runMap.begin();
  std::map<int,std::vector<int>*>::const_iterator runIend = runMap.end();
  while ( runIter != runIend ) {
    const std::pair<int,std::vector<int>*>& runEntry = *runIter++;
    // ----------- get full configuration
    int run = runEntry.first;
    std::vector<DTConfigKey> cfl;
    // ----------- get RH relation list
    std::vector<int>* rhlPtr = runEntry.second;
    std::vector<int>::const_iterator rhlIter = rhlPtr->begin();
    std::vector<int>::const_iterator rhlIend = rhlPtr->end();
    // ----------- loop over RH relations
    while ( rhlIter != rhlIend ) {
      int rhc = *rhlIter++;
      std::cout << "retrieve configuration bricks for run " << run
                << " ---> RH " << rhc << std::endl;
//      std::map<int,std::vector<DTConfigKey>*>::const_iterator
//               rhcIter = rhcMap.find( rhc );
//      std::map<int,std::vector<DTConfigKey>*>::const_iterator
//               rhcIend = rhcMap.end();
      std::map<int,std::vector<DTConfigKey>*>::const_iterator rhcIter;
      // ----------- redundant check
      if ( ( rhcIter = rhcMap.find( rhc ) ) == rhcMap.end() ) continue;
      std::vector<DTConfigKey>* listPtr = rhcIter->second;
      // ----------- redundant check
      if ( listPtr == 0 ) continue;
      std::vector<DTConfigKey>::const_iterator bkiIter = listPtr->begin();
      std::vector<DTConfigKey>::const_iterator bkiIend = listPtr->end();
      while ( bkiIter != bkiIend ) cfl.push_back( *bkiIter++ );
    }
    if ( sameConfigList( cfl, lastKey ) ) continue;
    lastKey = cfl;
    DTCCBConfig* fullConf = new DTCCBConfig( dataTag );
    // set run and full configuration in payload
    fullConf->setStamp(   run );
    fullConf->setFullKey( cfl );

    std::vector<DTConfigKey>::const_iterator cfgIter = cfl.begin();
    std::vector<DTConfigKey>::const_iterator cfgIend = cfl.end();
    while ( cfgIter != cfgIend ) {
      const DTConfigKey& cfgEntry = *cfgIter++;
      int cft = cfgEntry.confType;
      int cfg = cfgEntry.confKey;

      // =========== retrieve ccb config map
//    std::map<int,std::map<int,int>*>::const_iterator keyIter =
//                                                     keyMap.find( cfg );
//    std::map<int,std::map<int,int>*>::const_iterator keyIend =
//                                                     keyMap.end();
      std::map<int,int>* mapPtr = 0;
      std::map<int,std::map<int,int>*>::const_iterator keyIter;
      // ----------- redundant check
//      if ( keyIter != keyIend )
      if ( ( keyIter = keyMap.find( cfg ) ) != keyMap.end() )
           mapPtr = keyIter->second;
      if ( mapPtr == 0 ) continue;
      std::map<int,int>::const_iterator ccmIter = mapPtr->begin();
      std::map<int,int>::const_iterator ccmIend = mapPtr->end();
      while ( ccmIter != ccmIend ) {
        const std::pair<int,int>& ccmEntry = *ccmIter++;
        // ----------- get ccb config key
        int ccb = ccmEntry.first;
        int key = ccmEntry.second;
        // retrieve chamber id
//        std::map<int,DTCCBId>::const_iterator ccbIter = ccbMap.find( ccb );
//        std::map<int,DTCCBId>::const_iterator ccbIend = ccbMap.end();
        std::map<int,DTCCBId>::const_iterator ccbIter;
        // ----------- redundant check
//        if ( ccbIter == ccbIend ) continue;
        if ( ( ccbIter = ccbMap.find( ccb ) ) == ccbMap.end() ) continue;
        const DTCCBId& chaId = ccbIter->second;
        // ----------- retrieve brick id list
//        std::map<int,std::vector<int>*>::const_iterator brkIter =
//                                                        brkMap.find( key );
//        std::map<int,std::vector<int>*>::const_iterator brkIend =
//                                                        brkMap.end();
//        if ( brkIter == brkIend ) continue;
        std::map<int,std::vector<int>*>::const_iterator brkIter;
        if ( ( brkIter = brkMap.find( key ) ) == brkMap.end() ) continue;
        std::vector<int>* brkPtr = brkIter->second;
        // ----------- redundant check
        if ( brkPtr == 0 ) continue;
        // ----------- set brick id lists in payload
        std::vector<int> bkList;
        bkList.reserve( 20 );
//        std::map<int,int>::const_iterator bktIter = bktMap.begin();
//        std::map<int,int>::const_iterator bktIend = bktMap.end();
        std::map<int,int>::const_iterator bktIter;
        std::vector<int>::const_iterator bkiIter = brkPtr->begin();
        std::vector<int>::const_iterator bkiIend = brkPtr->end();
        while ( bkiIter != bkiIend ) {
          int brickId = *bkiIter++;
//          bktIter = bktMap.find( brickId );
          // ----------- redundant check
          if ( ( bktIter = bktMap.find( brickId ) ) == bktMap.end() ) continue;
          if ( bktIter->second == cft ) bkList.push_back( brickId );
        }
        fullConf->appendConfigKey( chaId.wheelId,
                                   chaId.stationId,
                                   chaId.sectorId,
                                   bkList );
      }
    }
    cond::Time_t snc = runEntry.first;
    m_to_transfer.push_back( std::make_pair( fullConf, snc ) );
    std::cout << "writing payload : " << sizeof( *fullConf ) 
              << " ( " << ( fullConf->end() - fullConf->begin() )
              << " ) " << std::endl;
  }

  isession.transaction().commit();
  isession.close();

  return;

}


void DTKeyedConfigHandler::chkConfigList() {

  std::cout << "open POOL out db " << std::endl;
  edm::Service<cond::service::PoolDBOutputService> outdb;

  std::cout << "start queries " << std::endl;
  std::map<int,bool> activeConfigMap;
  coral::ITable& fullConfigTable =
    isession.nominalSchema().tableHandle( "CONFIGSETS" );
  std::auto_ptr<coral::IQuery>
    fullConfigQuery( fullConfigTable.newQuery() );
  fullConfigQuery->addToOutputList( "CONFKEY" );
  fullConfigQuery->addToOutputList( "NAME" );
  fullConfigQuery->addToOutputList( "RUN" );
  coral::ICursor& fullConfigCursor = fullConfigQuery->execute();
  while( fullConfigCursor.next() ) {
    const coral::AttributeList& row = fullConfigCursor.currentRow();
    int fullConfigId = row["CONFKEY"].data<int>();
    int fullConfigRN = row["RUN"    ].data<int>();
    if ( fullConfigRN ) activeConfigMap.insert(
                        std::pair<int,bool>( fullConfigId, true ) );
    else                activeConfigMap.insert(
                        std::pair<int,bool>( fullConfigId, false ) );
    std::string fullConfigName = row["NAME"].data<std::string>();
  }

//  std::cout << " =============== CCB config list" << std::endl;
  std::map<int,bool> activeCCBCfgMap;
  coral::ITable& fullCCBCfgTable =
    isession.nominalSchema().tableHandle( "CCBRELATIONS" );
  std::auto_ptr<coral::IQuery>
    fullCCBCfgQuery( fullCCBCfgTable.newQuery() );
  fullCCBCfgQuery->addToOutputList( "CONFKEY" );
  fullCCBCfgQuery->addToOutputList( "CONFCCBKEY" );
  coral::ICursor& fullCCBCfgCursor = fullCCBCfgQuery->execute();
  while( fullCCBCfgCursor.next() ) {
    const coral::AttributeList& row = fullCCBCfgCursor.currentRow();
    int fullConfigId = row["CONFKEY"   ].data<int>();
    int fullCCBCfgId = row["CONFCCBKEY"].data<int>();
//    std::map<int,bool>::const_iterator cfgIter =
//                                       activeConfigMap.find( fullConfigId );
    std::map<int,bool>::const_iterator cfgIter;
    if ( ( cfgIter = activeConfigMap.find( fullConfigId ) )
                  == activeConfigMap.end() ) continue;
    if ( !( cfgIter->second ) ) continue;
    if ( activeCCBCfgMap.find( fullCCBCfgId ) ==
         activeCCBCfgMap.end() ) 
         activeCCBCfgMap.insert( std::pair<int,bool>( fullCCBCfgId, true ) );
  }

//  std::cout << " =============== config brick list" << std::endl;
  std::map<int,bool> activeCfgBrkMap;
  coral::ITable& ccbConfBrkTable =
    isession.nominalSchema().tableHandle( "CFG2BRKREL" );
  std::auto_ptr<coral::IQuery>
    ccbConfBrickQuery( ccbConfBrkTable.newQuery() );
  ccbConfBrickQuery->addToOutputList( "CONFID" );
  ccbConfBrickQuery->addToOutputList( "BRKID" );
  coral::ICursor& ccbConfBrickCursor = ccbConfBrickQuery->execute();
  while( ccbConfBrickCursor.next() ) {
    const coral::AttributeList& row = ccbConfBrickCursor.currentRow();
    int fullCCBCfgId = row["CONFID"].data<int>();
    int ccbConfBrkId = row["BRKID" ].data<int>();
//    std::map<int,bool>::const_iterator ccbIter =
//                                       activeCCBCfgMap.find( fullCCBCfgId );
    std::map<int,bool>::const_iterator ccbIter;
    if ( ( ccbIter = activeCCBCfgMap.find( fullCCBCfgId ) )
                  == activeCCBCfgMap.end() ) continue;
    if ( !( ccbIter->second ) ) continue;
    if ( activeCfgBrkMap.find( ccbConfBrkId ) ==
         activeCfgBrkMap.end() )
         activeCfgBrkMap.insert( std::pair<int,bool>( ccbConfBrkId, true ) );
  }

//  std::cout << " ===============" << std::endl;

  coral::ITable& brickConfigTable =
    isession.nominalSchema().tableHandle( "CFGBRICKS" );
  std::auto_ptr<coral::IQuery>
    brickConfigQuery( brickConfigTable.newQuery() );
  brickConfigQuery->addToOutputList( "BRKID" );
  brickConfigQuery->addToOutputList( "BRKNAME" );
  coral::ICursor& brickConfigCursor = brickConfigQuery->execute();
  DTKeyedConfig* brickData = 0;
  std::vector<int> missingList;
  std::vector<unsigned long long> checkedKeys;
  while( brickConfigCursor.next() ) {
    const coral::AttributeList& row = brickConfigCursor.currentRow();
    int brickConfigId = row["BRKID"].data<int>();
    if ( brickConfigId < minBrickId ) continue;
    if ( brickConfigId > maxBrickId ) continue;
//    std::map<int,bool>::const_iterator brkIter =
//                                       activeCfgBrkMap.find( brickConfigId );
    std::map<int,bool>::const_iterator brkIter;
    if ( ( brkIter = activeCfgBrkMap.find( brickConfigId ) )
                  == activeCfgBrkMap.end() ) continue;
    if ( !( brkIter->second ) ) continue;
    std::string brickConfigName = row["BRKNAME"].data<std::string>();
    std::cout << "brick " << brickConfigId
              << " : "    << brickConfigName << std::endl;
    checkedKeys.push_back( brickConfigId );
    bool brickFound = false;
    try {
      keyList->load( checkedKeys );
      boost::shared_ptr<DTKeyedConfig> brickCheck = keyList->get<DTKeyedConfig>( 0 );
      if ( brickCheck.get() ) brickFound =
                             ( brickCheck->getId() == brickConfigId );
    }
    catch ( std::exception e ) {
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
//    std::cout << "get data for brick: " << brickConfigId << std::endl;
    coral::AttributeList bindVariableList;
    bindVariableList.extend( "brickId", typeid(int) );
    bindVariableList["brickId"].data<int>() = brickConfigId;
    std::auto_ptr<coral::IQuery>
           brickDataQuery( isession.nominalSchema().newQuery() );
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


std::string DTKeyedConfigHandler::id() const {
  return dataTag;
}


bool DTKeyedConfigHandler::sameConfigList(
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


void DTKeyedConfigHandler::setList( cond::persistency::KeyList* list ) {
  keyList = list;
}

