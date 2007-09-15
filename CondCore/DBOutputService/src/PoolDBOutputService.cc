#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ConfigSessionFromParameterSet.h"
#include "CondCore/DBOutputService/interface/Exception.h"
#include "CondCore/DBCommon/interface/ObjectRelationalMappingUtility.h"
#include "CondCore/DBCommon/interface/DBSession.h"
//POOL include
#include "FileCatalog/IFileCatalog.h"
#include "serviceCallbackToken.h"
//#include <iostream>
#include <vector>
cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ): 
  m_currentTime( 0 ),
  m_session( 0 ),
  m_dbstarted( false )
{
  std::string connect=iConfig.getParameter<std::string>("connect");
  m_session=new cond::DBSession;  
  m_timetype=iConfig.getParameter< std::string >("timetype");
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters"); 
  ConfigSessionFromParameterSet configConnection(*m_session,connectionPset);
  static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
  std::string catconnect("pfncatalog_memory://POOL_RDBMS?");
  catconnect.append(connect);
  conHandler.registerConnection("outputdb",connect,0);
  m_session->open();
  conHandler.connect(m_session);
  m_connection=conHandler.getConnection("outputdb");
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toPut=iConfig.getParameter<Parameters>("toPut");
  for(Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut) {
    cond::service::serviceCallbackRecord thisrecord;
    thisrecord.m_containerName = itToPut->getParameter<std::string>("record");
    thisrecord.m_tag = itToPut->getParameter<std::string>("tag");
    m_callbacks.insert(std::make_pair(cond::service::serviceCallbackToken::build(thisrecord.m_containerName),thisrecord));
  }
  iAR.watchPreProcessEvent(this,&cond::service::PoolDBOutputService::preEventProcessing);
  iAR.watchPostEndJob(this,&cond::service::PoolDBOutputService::postEndJob);
  iAR.watchPreModule(this,&cond::service::PoolDBOutputService::preModule);
  iAR.watchPostModule(this,&cond::service::PoolDBOutputService::postModule);
}

std::string 
cond::service::PoolDBOutputService::tag( const std::string& EventSetupRecordName ){
  return this->lookUpRecord(EventSetupRecordName).m_tag;
}

bool 
cond::service::PoolDBOutputService::isNewTagRequest( const std::string& EventSetupRecordName ){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if(!m_dbstarted) this->initDB();
  return myrecord.m_isNewTag;
}
void 
cond::service::PoolDBOutputService::initDB()
{
  if(m_dbstarted) return;
  cond::CoralTransaction& coraldb=m_connection->coralTransaction(false);
  try{
    coraldb.start(); 
    cond::ObjectRelationalMappingUtility* mappingUtil=new cond::ObjectRelationalMappingUtility(&(coraldb.coralSessionProxy()) );
    if( !mappingUtil->existsMapping(cond::IOVNames::iovMappingVersion()) ){
      mappingUtil->buildAndStoreMappingFromBuffer(cond::IOVNames::iovMappingXML());
    }
    delete mappingUtil;
    cond::MetaData metadata(coraldb);
    for(std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.begin(); it!=m_callbacks.end(); ++it){
      //std::string iovtoken;
      if( !metadata.hasTag(it->second.m_tag) ){
	it->second.m_iovtoken="";
	it->second.m_isNewTag=true;
      }else{
	it->second.m_iovtoken=metadata.getToken(it->second.m_tag);
	it->second.m_isNewTag=false;
      }
    }
    coraldb.commit();
  }catch( const cond::Exception& er ){
    coraldb.rollback();
    throw cms::Exception( er.what() );
  }catch( const std::exception& er ){
    coraldb.rollback();
    throw cms::Exception( er.what() );
  }
  m_dbstarted=true;
}

void 
cond::service::PoolDBOutputService::postEndJob()
{

}

void 
cond::service::PoolDBOutputService::preEventProcessing(const edm::EventID& iEvtid, const edm::Timestamp& iTime)
{
  if( m_timetype == "runnumber" ){
    m_currentTime=iEvtid.run();
  }else{ //timestamp
    m_currentTime=iTime.value();
  }
}
void
cond::service::PoolDBOutputService::preModule(const edm::ModuleDescription& desc){
}

void
cond::service::PoolDBOutputService::postModule(const edm::ModuleDescription& desc){
}

cond::service::PoolDBOutputService::~PoolDBOutputService(){
  delete m_session;
}

size_t 
cond::service::PoolDBOutputService::callbackToken(const std::string& EventSetupRecordName ) const {
  return cond::service::serviceCallbackToken::build(EventSetupRecordName);
}

cond::Time_t 
cond::service::PoolDBOutputService::endOfTime() const{
  return 1234;
}

cond::Time_t 
cond::service::PoolDBOutputService::currentTime() const{
  return m_currentTime;
}

void 
cond::service::PoolDBOutputService::createNewIOV( const std::string& firstPayloadToken, cond::Time_t firstTillTime,const std::string& EventSetupRecordName){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if (!m_dbstarted) this->initDB();
  if(!myrecord.m_isNewTag) throw cond::Exception("not a new tag");
  cond::PoolTransaction& pooldb=m_connection->poolTransaction(false);
  std::string iovToken;
  try{
    pooldb.start();
    iovToken=this->insertIOV(pooldb,myrecord,firstPayloadToken,firstTillTime);
    pooldb.commit();
  }catch(...){
    pooldb.rollback();
    throw cond::Exception("PoolDBOutputService::createNewIOV error in commit");
  }
  cond::CoralTransaction& coraldb=m_connection->coralTransaction(false);
  try{
    cond::MetaData metadata(coraldb);
    coraldb.start();
    metadata.addMapping(myrecord.m_tag,iovToken);
    coraldb.commit();
  }catch(const std::exception&){
    //I hope it'll never happen!
    coraldb.rollback();
    //TODO: proper cleaning and rollback  in sequence
    //delete new payload; 
    //pooldb.start();
    //this->resetPreviousPoolStatus(payloadToken,iovToken);
    //pooldb.commit();
  }
  m_newtags.push_back( std::make_pair<std::string,std::string>(myrecord.m_tag,iovToken) );
  myrecord.m_iovtoken=iovToken;
  myrecord.m_isNewTag=false;
}

void 
cond::service::PoolDBOutputService::appendTillTime( const std::string& payloadToken, cond::Time_t tillTime,const std::string& EventSetupRecordName){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if (!m_dbstarted) this->initDB();
  cond::PoolTransaction& pooldb=m_connection->poolTransaction(false);
  try{
    pooldb.start();
    this->insertIOV(pooldb,myrecord,payloadToken,tillTime);
    pooldb.commit();
  }catch(const std::exception&){
    pooldb.rollback();
    throw cond::Exception("PoolDBOutputService::appendTillTime error in commit");
  }
}

void 
cond::service::PoolDBOutputService::appendSinceTime( const std::string& payloadToken, cond::Time_t sinceTime,const std::string& EventSetupRecordName ){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if (!m_dbstarted) this->initDB();
  cond::PoolTransaction& pooldb=m_connection->poolTransaction(false);
  try{
    pooldb.start();
    this->appendIOV(pooldb,myrecord,payloadToken,sinceTime);
    pooldb.commit();
  }catch(const std::exception&){
    pooldb.rollback();
    throw cond::Exception("PoolDBOutputService::appendSinceTime error in commit");
  }
}
cond::service::serviceCallbackRecord& 
cond::service::PoolDBOutputService::lookUpRecord(const std::string& EventSetupRecordName){
  size_t callbackToken=this->callbackToken( EventSetupRecordName );
  std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.find(callbackToken);
  if(it==m_callbacks.end()) throw cond::UnregisteredRecordException(EventSetupRecordName);
  return it->second;
}

void 
cond::service::PoolDBOutputService::appendIOV(cond::PoolTransaction& pooldb,
						   cond::service::serviceCallbackRecord& record, 
						   const std::string& payloadToken, 
						   cond::Time_t sinceTime){
  if( record.m_isNewTag ) {
    throw cond::Exception(std::string("PoolDBOutputService::appendIOV: cannot append to non-existing tag ")+record.m_tag );  
  }
  cond::IOVService iovmanager(pooldb);  
  cond::IOVEditor* editor=iovmanager.newIOVEditor(record.m_iovtoken);
  editor->append(sinceTime,payloadToken);
  delete editor;
}
std::string 
cond::service::PoolDBOutputService::insertIOV( cond::PoolTransaction& pooldb,
					       cond::service::serviceCallbackRecord& record, 
					       const std::string& payloadToken,
					       cond::Time_t tillTime)
{
  cond::IOVService iovmanager(pooldb);
  cond::IOVEditor* editor=iovmanager.newIOVEditor();
  editor->insert(tillTime,payloadToken);
  std::string iovToken=editor->token();
  delete editor;    
  return iovToken;
}
