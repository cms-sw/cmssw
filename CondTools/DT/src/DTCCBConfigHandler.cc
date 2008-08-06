/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/06/03 16:25:04 $
 *  $Revision: 1.7 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTCCBConfigHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTConfigHandler.h"
#include "CondTools/DT/interface/DTDBSession.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DTObjects/interface/DTConfigList.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "RelationalAccess/ISessionProxy.h"
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


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTCCBConfigHandler::DTCCBConfigHandler( const edm::ParameterSet& ps ):
 dataTag(               ps.getParameter<std::string> ( "tag" ) ),
 onlineConnect(         ps.getParameter<std::string> ( "onlineDB" ) ),
 onlineAuthentication(  ps.getParameter<std::string> ( 
                        "onlineAuthentication" ) ),
 offlineAuthentication( ps.getParameter<edm::ParameterSet>( "DBParameters" )
                          .getUntrackedParameter<std::string> (
                        "authenticationPath" ) ),
 offlineConnect(        ps.getParameter<std::string> ( "offlineDB" ) ),
// catalog(               ps.getParameter<std::string> ( "catalog" ) ),
 listToken(             ps.getParameter<std::string> ( "token" ) ) {
  std::cout << " PopCon application for DT configuration export "
            <<  onlineAuthentication << " "
            << offlineAuthentication << std::endl;
}

//--------------
// Destructor --
//--------------
DTCCBConfigHandler::~DTCCBConfigHandler() {
}

//--------------
// Operations --
//--------------
void DTCCBConfigHandler::getNewObjects() {

  cond::DBSession* coralSession;
  cond::Connection* m_connection;
  cond::CoralTransaction* m_coraldb;

  coralSession = new cond::DBSession();
  // to get the username/passwd from $CORAL_AUTH_PATH/authentication.xml
  coralSession->configuration().setAuthenticationMethod( cond::XML );
  coralSession->configuration().setAuthenticationPath( onlineAuthentication );
  // set message level to Error or Debug
  coralSession->configuration().setMessageLevel( cond::Error );
//  coralSession->connectionConfiguration().setConnectionRetrialTimeOut( 60 );
  coralSession->open();

  m_connection = new cond::Connection( onlineConnect );
  m_connection->connect( coralSession );
  m_coraldb = &( m_connection->coralTransaction() );
  m_coraldb->start( true );

//  std::cout << "get session proxy... " << std::endl;
  isession = &( m_coraldb->coralSessionProxy() );
//  std::cout << "session proxy got" << std::endl;
  m_coraldb->start( true );

  chkConfigList();
//  std::cout << "get run config..." << std::endl;

  //to access the information on the tag inside the offline database:
  cond::TagInfo const & ti = tagInfo();
  unsigned int last = ti.lastInterval.first;
  std::cout << "last configuration key already copied for run: "
            << last << std::endl;

  if ( last == 0 ) {
    DTCCBConfig* dummyConf = new DTCCBConfig( dataTag );
    dummyConf->setStamp( 0 );
    dummyConf->setFullKey( 0 );
    cond::Time_t snc = 1;
    m_to_transfer.push_back( std::make_pair( dummyConf, snc ) );
  }

  //to access the information on last successful log entry for this tag:
//  cond::LogDBEntry const & lde = logDBEntry();     

  //to access the lastest payload (Ref is a smart pointer)
//  Ref payload = lastPayload();

  unsigned lastRun = last;
  std::cout << "check for new runs since " << lastRun << std::endl;

  // Find latest runs
  std::map<int,int> g2lMap;
  std::map<int,int> runMap;
  coral::ITable& runHistoryTable =
    isession->nominalSchema().tableHandle( "RUNHISTORY" );
  std::auto_ptr<coral::IQuery>
    runHistoryQuery( runHistoryTable.newQuery() );
  runHistoryQuery->addToOutputList( "RUN" );
  runHistoryQuery->addToOutputList( "G2LID" );
  coral::ICursor& runHistoryCursor = runHistoryQuery->execute();
  while( runHistoryCursor.next() ) {
    const coral::AttributeList& row = runHistoryCursor.currentRow();
    int runId = row[  "RUN"].data<int>();
    int g2lId = row["G2LID"].data<int>();
    if ( static_cast<unsigned>( runId ) <= lastRun ) continue;
    std::cout << "schedule config key copy for run "
              << runId << std::endl;
    if ( g2lMap.find( g2lId ) == g2lMap.end() )
         g2lMap.insert( std::pair<int,int>( g2lId, runId ) );
    if ( runMap.find( runId ) == runMap.end() )
         runMap.insert( std::pair<int,int>( runId, g2lId ) );
  }
  if ( !runMap.size() ) std::cout << "no new run found" << std::endl;

  // get full config for latest runs
  std::map<int,int> l2gMap;
  std::map<int,int> cfgMap;
  coral::ITable& runConfigTable =
    isession->nominalSchema().tableHandle( "G2LRELATIONS" );
  std::auto_ptr<coral::IQuery>
    runConfigQuery( runConfigTable.newQuery() );
  runConfigQuery->addToOutputList( "ID" );
  runConfigQuery->addToOutputList( "CCBCSET" );
  coral::ICursor& runConfigCursor = runConfigQuery->execute();
  while( runConfigCursor.next() ) {
    const coral::AttributeList& row = runConfigCursor.currentRow();
    int g2lId = row[     "ID"].data<int>();
    int cfgId = row["CCBCSET"].data<int>();
    std::map<int,int>::const_iterator g2lIter = g2lMap.find( g2lId );
    std::map<int,int>::const_iterator g2lIend = g2lMap.end();
    if ( g2lIter == g2lIend ) continue;
    l2gMap.insert( std::pair<int,int>( g2lId, cfgId ) );
    if ( cfgMap.find( cfgId ) == cfgMap.end() )
    cfgMap.insert( std::pair<int,int>( cfgId, g2lId ) );
  }

  // get ccb identifiers map
  std::cout << "retrieve CCB map" << std::endl;
  std::map<int,DTCCBId> ccbMap;
  coral::ITable& ccbMapTable =
    isession->nominalSchema().tableHandle( "CCBMAP" );
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

  // get ccb config keys
  std::cout << "retrieve CCB configuration keys" << std::endl;
  std::map<int,std::map<int,int>*> keyMap;
  std::map<int,int> cckMap;
  coral::ITable& ccbRelTable =
    isession->nominalSchema().tableHandle( "CCBRELATIONS" );
  std::auto_ptr<coral::IQuery>
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
    if ( cfgMap.find( cfg ) == cfgMap.end() ) continue;
    std::map<int,std::map<int,int>*>::const_iterator keyIter =
                                                     keyMap.find( cfg );
    std::map<int,std::map<int,int>*>::const_iterator keyIend =
                                                     keyMap.end();
    std::map<int,int>* mapPtr = 0;
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
  std::auto_ptr<coral::IQuery>
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
    std::vector<int>* brkPtr = 0;
    if ( brkIter != brkIend ) brkPtr = brkIter->second;
    else                      brkMap.insert(
                              std::pair<int,std::vector<int>*>( key,
                              brkPtr = new std::vector<int> ) );
    // store brick key
    brkPtr->push_back( brk );
  }

  // loop over new runs
  std::map<int,int>::const_iterator runIter = runMap.begin();
  std::map<int,int>::const_iterator runIend = runMap.end();
  while ( runIter != runIend ) {
    const std::pair<int,int>& runEntry = *runIter++;
    // get full configuration
    int run = runEntry.first;
    int g2l = runEntry.second;
    // retrieve full configuration
    std::map<int,int>::const_iterator l2gIter = l2gMap.find( g2l );
    std::map<int,int>::const_iterator l2gIend = l2gMap.end();
    if ( l2gIter == l2gIend ) continue;
    const std::pair<int,int>& l2gEntry = *l2gIter;
    int cfg = l2gEntry.second;
    std::cout << "retrieve configuration bricks for run " << run
              << " ---> g2l_id " << g2l
              << " ---> config " << cfg << std::endl;
    DTCCBConfig* fullConf = new DTCCBConfig( dataTag );
    // set run and full configuration in payload
    fullConf->setStamp(   run );
    fullConf->setFullKey( cfg );
    // retrieve ccb config map
    std::map<int,std::map<int,int>*>::const_iterator keyIter =
                                                     keyMap.find( cfg );
    std::map<int,std::map<int,int>*>::const_iterator keyIend =
                                                     keyMap.end();
    std::map<int,int>* mapPtr = 0;
    if ( keyIter != keyIend ) mapPtr = keyIter->second;
    if ( mapPtr == 0 ) continue;
    // loop over ccb
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
      if ( brkPtr == 0 ) continue;
      // brick id lists in payload
      fullConf->setConfigKey( chaId.wheelId,
                              chaId.stationId,
                              chaId.sectorId,
                              *brkPtr );
    }
    cond::Time_t snc = runEntry.first;
    m_to_transfer.push_back( std::make_pair( fullConf, snc ) );
    std::cout << "writing payload : " << sizeof( *fullConf ) 
              << " ( " << ( fullConf->end() - fullConf->begin() )
              << " ) " << std::endl;
  }

  delete m_connection;
  delete coralSession;

  return;

}


void DTCCBConfigHandler::chkConfigList() {
  std::cout << "check for available configurations " << std::endl
            << "create session: "
            << offlineConnect << " , "
            << offlineCatalog << " , "
            << offlineAuthentication << std::endl;

  DTDBSession* session = new DTDBSession( offlineConnect, offlineCatalog,
                                          offlineAuthentication );
  session->connect( false );
  const DTConfigList* rs;
//  std::cout << "read full list: " << listToken << std::endl;
  DTConfigHandler* ri = DTConfigHandler::create( session, listToken );
  rs = ri->getContainer();

  std::map<int,bool> activeConfigMap;
  coral::ITable& fullConfigTable =
    isession->nominalSchema().tableHandle( "CONFIGSETS" );
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
//    std::cout << "config " << fullConfigId
//              << " : "     << fullConfigName << std::endl;
  }

//  std::cout << " =============== CCB config list" << std::endl;
  std::map<int,bool> activeCCBCfgMap;
  coral::ITable& fullCCBCfgTable =
    isession->nominalSchema().tableHandle( "CCBRELATIONS" );
  std::auto_ptr<coral::IQuery>
    fullCCBCfgQuery( fullCCBCfgTable.newQuery() );
  fullCCBCfgQuery->addToOutputList( "CONFKEY" );
  fullCCBCfgQuery->addToOutputList( "CONFCCBKEY" );
  coral::ICursor& fullCCBCfgCursor = fullCCBCfgQuery->execute();
  while( fullCCBCfgCursor.next() ) {
    const coral::AttributeList& row = fullCCBCfgCursor.currentRow();
    int fullConfigId = row["CONFKEY"   ].data<int>();
    int fullCCBCfgId = row["CONFCCBKEY"].data<int>();
    std::map<int,bool>::const_iterator cfgIter =
                                       activeConfigMap.find( fullConfigId );
    if ( cfgIter == activeConfigMap.end() ) continue;
    if ( activeCCBCfgMap.find( fullCCBCfgId ) ==
         activeCCBCfgMap.end() ) 
         activeCCBCfgMap.insert( std::pair<int,bool>( fullCCBCfgId, true ) );
  }

//  std::cout << " =============== config brick list" << std::endl;
  std::map<int,bool> activeCfgBrkMap;
  coral::ITable& ccbConfBrkTable =
    isession->nominalSchema().tableHandle( "CFG2BRKREL" );
  std::auto_ptr<coral::IQuery>
    ccbConfBrickQuery( ccbConfBrkTable.newQuery() );
  ccbConfBrickQuery->addToOutputList( "CONFID" );
  ccbConfBrickQuery->addToOutputList( "BRKID" );
  coral::ICursor& ccbConfBrickCursor = ccbConfBrickQuery->execute();
  while( ccbConfBrickCursor.next() ) {
    const coral::AttributeList& row = ccbConfBrickCursor.currentRow();
    int fullCCBCfgId = row["CONFID"].data<int>();
    int ccbConfBrkId = row["BRKID" ].data<int>();
    std::map<int,bool>::const_iterator ccbIter =
                                       activeCCBCfgMap.find( fullCCBCfgId );
    if ( ccbIter == activeCCBCfgMap.end() ) continue;
    if ( !( ccbIter->second ) ) continue;
    if ( activeCfgBrkMap.find( ccbConfBrkId ) ==
         activeCfgBrkMap.end() )
         activeCfgBrkMap.insert( std::pair<int,bool>( ccbConfBrkId, true ) );
  }

//  std::cout << " ===============" << std::endl;

  coral::ITable& brickConfigTable =
    isession->nominalSchema().tableHandle( "CFGBRICKS" );
  std::auto_ptr<coral::IQuery>
    brickConfigQuery( brickConfigTable.newQuery() );
  brickConfigQuery->addToOutputList( "BRKID" );
  brickConfigQuery->addToOutputList( "BRKNAME" );
  coral::ICursor& brickConfigCursor = brickConfigQuery->execute();
  DTConfigData* brickData = 0;
//  char brickIdString[20];
  std::vector<int> missingList;
  while( brickConfigCursor.next() ) {
    const coral::AttributeList& row = brickConfigCursor.currentRow();
    int brickConfigId = row["BRKID"].data<int>();
    std::map<int,bool>::const_iterator brkIter =
                                       activeCfgBrkMap.find( brickConfigId );
    if ( brkIter == activeCfgBrkMap.end() ) continue;
    if ( !( brkIter->second ) ) continue;
    std::string brickConfigName = row["BRKNAME"].data<std::string>();
//    std::cout << "brick " << brickConfigId
//              << " : "    << brickConfigName << std::endl;
    ri->get( brickConfigId, brickData );
    if ( brickData == 0 ) {
      std::cout << "brick " << brickConfigId << " missing, copy request"
                << std::endl;
      missingList.push_back( brickConfigId );
    }
  }

  std::vector<int>::const_iterator brickIter = missingList.begin();
  std::vector<int>::const_iterator brickIend = missingList.end();
  while ( brickIter != brickIend ) {
    int brickConfigId = *brickIter++;
    coral::AttributeList bindVariableList;
    bindVariableList.extend( "brickId", typeid(int) );
    bindVariableList["brickId"].data<int>() = brickConfigId;
    std::auto_ptr<coral::IQuery>
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
    brickData = new DTConfigData();
    brickData->setId( brickConfigId );
    while( brickDataCursor.next() ) {
      const coral::AttributeList& row = brickDataCursor.currentRow();
      brickData->add( row["CONFIGCMDS.CONFDATA"].data<std::string>() );
    }
    cond::TypedRef<DTConfigData> brickRef( *session->poolDB(), brickData );
    brickRef.markWrite( "DTConfigData" );
    ri->set( brickConfigId, brickRef.token() );
  }

  std::cout << "brick list updated, disconnect session" << std::endl;
//  std::cout << "disconnect session..." << std::endl;
  session->disconnect();
//  std::cout << "delete session..." << std::endl;
  delete session;
//  std::cout << "list updated..." << std::endl;

  return;

}


std::string DTCCBConfigHandler::id() const {
  return dataTag;
}


