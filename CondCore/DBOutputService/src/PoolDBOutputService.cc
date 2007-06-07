#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
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
#include "CondCore/DBCommon/interface/DBCatalog.h"
//POOL include
#include "FileCatalog/IFileCatalog.h"
#include "serviceCallbackToken.h"
//#include <iostream>
#include <vector>
cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ): 
  m_currentTime( 0 ),
  m_session( 0 ),
  m_iovservice( 0 ),
  m_pooldb( 0 ),
  m_coraldb( 0 ),
  m_dbstarted( false ),
  m_inTransaction( false )
{
  std::string connect=iConfig.getParameter<std::string>("connect");
  std::string catconnect=iConfig.getUntrackedParameter<std::string>("catalog","");
  cond::DBCatalog mycat;
  std::pair<std::string,std::string> logicalService=mycat.logicalservice(connect);
  std::string logicalServiceName=logicalService.first;
  std::string dataName=logicalService.second;
  if( !logicalServiceName.empty() ){
    if( catconnect.empty() ){
      if( logicalServiceName=="dev" ){
	catconnect=mycat.defaultDevCatalogName();
      }else if( logicalServiceName=="online" ){
	catconnect=mycat.defaultOnlineCatalogName();
      }else if( logicalServiceName=="offline" ){
	catconnect=mycat.defaultOfflineCatalogName();
      }else if( logicalServiceName=="local" ){
	catconnect="file:localCondDBCatalog.xml";
	connect=std::string("sqlite_file:")+dataName;
      }else{
	throw cond::Exception(std::string("no default catalog found for ")+logicalServiceName);
      }
    }
    if(logicalServiceName != "local"){
      mycat.poolCatalog().setWriteCatalog(catconnect);
      mycat.poolCatalog().connect();
      mycat.poolCatalog().start();
      std::string pf=mycat.getPFN(mycat.poolCatalog(),connect, false);
      mycat.poolCatalog().commit();
      mycat.poolCatalog().disconnect();
      connect=pf;
    }
  }
  m_session=new cond::DBSession(true);
  std::string timetype=iConfig.getParameter< std::string >("timetype");
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters"); 
  ConfigSessionFromParameterSet configConnection(*m_session,connectionPset);
  m_session->open();
  m_pooldb=new cond::PoolStorageManager(connect,catconnect,m_session);
  m_coraldb=new cond::RelationalStorageManager(connect,m_session);
  if( timetype=="timestamp" ){
    m_iovservice=new cond::IOVService(*m_pooldb,cond::timestamp);
  }else{
    m_iovservice=new cond::IOVService(*m_pooldb,cond::runnumber);
  }
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

std::string cond::service::PoolDBOutputService::tag( const std::string& EventSetupRecordName ){
  return this->lookUpRecord(EventSetupRecordName).m_tag;
}
bool cond::service::PoolDBOutputService::isNewTagRequest( const std::string& EventSetupRecordName ){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if(!m_dbstarted) this->initDB();
  return myrecord.m_isNewTag;
}
void 
cond::service::PoolDBOutputService::initDB()
{
  if(m_dbstarted) return;
  try{
    m_coraldb->connect(cond::ReadWriteCreate);
    cond::ObjectRelationalMappingUtility* mappingUtil=new cond::ObjectRelationalMappingUtility(*m_coraldb);
    m_coraldb->startTransaction(false); 
    if( !mappingUtil->existsMapping(cond::IOVNames::iovMappingVersion()) ){
      mappingUtil->buildAndStoreMappingFromBuffer(cond::IOVNames::iovMappingXML());
    }
    m_coraldb->commit();
    delete mappingUtil;
    cond::MetaData metadata(*m_coraldb);
    m_coraldb->startTransaction(true);
    for(std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.begin(); it!=m_callbacks.end(); ++it){
      std::string iovtoken;
      if( !metadata.hasTag(it->second.m_tag) ){
	it->second.m_isNewTag=true;
      }else{
	iovtoken=metadata.getToken(it->second.m_tag);
	it->second.m_isNewTag=false;
      }
      it->second.m_iovEditor=m_iovservice->newIOVEditor(iovtoken);
    }
    m_coraldb->commit();
    m_coraldb->disconnect();
    m_pooldb->connect();
  }catch( const cond::Exception& er ){
    throw cms::Exception( er.what() );
  }catch( const std::exception& er ){
    throw cms::Exception( er.what() );
  }catch(...){
    throw cms::Exception( "Unknown error" );
  }
  m_dbstarted=true;
}

void 
cond::service::PoolDBOutputService::postEndJob()
{
  if(!m_dbstarted) return;
  if(m_inTransaction){
    m_pooldb->commit();
    m_inTransaction=false;
  }
  m_pooldb->disconnect();
  m_coraldb->connect(cond::ReadWriteCreate);
  m_coraldb->startTransaction(false); 
  cond::MetaData metadata(*m_coraldb);
  for(std::vector<std::pair<std::string,std::string> >::iterator it=m_newtags.begin(); it!=m_newtags.end(); ++it){
    //std::cout<<"adding "<<it->first<<" "<<it->second<<std::endl;
    metadata.addMapping(it->first,it->second);
  }
  m_coraldb->commit();
  m_coraldb->disconnect();
  m_session->close();
  m_dbstarted=false;
}
void 
cond::service::PoolDBOutputService::preEventProcessing(const edm::EventID& iEvtid, const edm::Timestamp& iTime)
{
  if( m_iovservice->timeType() == cond::runnumber ){
    m_currentTime=iEvtid.run();
  }else{ //timestamp
    m_currentTime=iTime.value();
  }
}
void
cond::service::PoolDBOutputService::preModule(const edm::ModuleDescription& desc){
  //std::cout<<"preModule"<<std::endl;
  if(m_dbstarted){
    m_pooldb->startTransaction(false);    
  }
}
void
cond::service::PoolDBOutputService::postModule(const edm::ModuleDescription& desc){
  //std::cout<<"postModule"<<std::endl;
}
cond::service::PoolDBOutputService::~PoolDBOutputService(){
  for(std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.begin(); it!=m_callbacks.end(); ++it){
    if(it->second.m_iovEditor){
      delete it->second.m_iovEditor;
      it->second.m_iovEditor=0;
    }
  }
  m_callbacks.clear();
  delete m_iovservice;
  delete m_session;
}
size_t cond::service::PoolDBOutputService::callbackToken(const std::string& EventSetupRecordName ) const {
  return cond::service::serviceCallbackToken::build(EventSetupRecordName);
}
cond::Time_t cond::service::PoolDBOutputService::endOfTime() const{
  return m_iovservice->globalTill();
}
cond::Time_t cond::service::PoolDBOutputService::currentTime() const{
  return m_currentTime;
}
void cond::service::PoolDBOutputService::createNewIOV( const std::string& firstPayloadToken, cond::Time_t firstTillTime,const std::string& EventSetupRecordName){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if (!m_dbstarted) this->initDB();
  if(!myrecord.m_isNewTag) throw cond::Exception("not a new tag");
  if ( !m_inTransaction ){
    m_pooldb->startTransaction(false);    
    m_inTransaction=true;
  }
  std::string iovToken=this->insertIOV(myrecord,firstPayloadToken,firstTillTime,EventSetupRecordName);
  m_newtags.push_back( std::make_pair<std::string,std::string>(myrecord.m_tag,iovToken) );
  myrecord.m_isNewTag=false;
}
void cond::service::PoolDBOutputService::appendTillTime( const std::string& payloadToken, cond::Time_t tillTime,const std::string& EventSetupRecordName){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if (!m_dbstarted) this->initDB();
  if ( !m_inTransaction ){
    m_pooldb->startTransaction(false);    
    m_inTransaction=true;
  }
  this->insertIOV(myrecord,payloadToken,tillTime,EventSetupRecordName);
}
void cond::service::PoolDBOutputService::appendSinceTime( const std::string& payloadToken, cond::Time_t sinceTime,const std::string& EventSetupRecordName ){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if (!m_dbstarted) this->initDB();
  if ( !m_inTransaction ){
    m_pooldb->startTransaction(false);    
    m_inTransaction=true;
  }
  this->appendIOV(myrecord,payloadToken,sinceTime);
}
void cond::service::PoolDBOutputService::appendIOV(cond::service::serviceCallbackRecord& record, const std::string& payloadToken, cond::Time_t sinceTime){
  if( record.m_isNewTag ) throw cond::Exception(std::string("PoolDBOutputService::appendIOV: cannot append to non-existing tag ")+record.m_tag );  
  if ( !m_inTransaction ){
    m_pooldb->startTransaction(false);    
    m_inTransaction=true;
  }
  record.m_iovEditor->append(sinceTime,payloadToken);
}
std::string cond::service::PoolDBOutputService::insertIOV(cond::service::serviceCallbackRecord& record, const std::string& payloadToken, cond::Time_t tillTime, const std::string& EventSetupRecordName){
  //std::cout<<"insertIOV payloadToken"<<payloadToken<<std::endl;
  //std::cout<<"tillTime "<<tillTime<<std::endl;
  //std::cout<<"record "<<EventSetupRecordName<<std::endl;
  record.m_iovEditor->insert(tillTime,payloadToken);
  std::string iovToken=record.m_iovEditor->token();
  return iovToken;
}
cond::service::serviceCallbackRecord& cond::service::PoolDBOutputService::lookUpRecord(const std::string& EventSetupRecordName){
  size_t callbackToken=this->callbackToken( EventSetupRecordName );
  std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.find(callbackToken);
  if(it==m_callbacks.end()) throw cond::UnregisteredRecordException(EventSetupRecordName);
  return it->second;
}
