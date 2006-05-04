#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataSvc/IDataSvc.h"
#include "DataSvc/ICacheSvc.h"

//#include "CondCore/IOVService/interface/IOV.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
//#include "CondCore/DBCommon/interface/DBSession.h"
//#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include <iostream>
#include <exception>
cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ): 
  m_connect( iConfig.getParameter< std::string > ("connect") ),
  m_tag( iConfig.getParameter< std::string >("tag") ),
  m_timetype( iConfig.getParameter< std::string >("timetype") ),
  m_clientmodule(iConfig.getUntrackedParameter< std::string >("moduleToWatch") ), 
  m_connectMode( iConfig.getUntrackedParameter< unsigned int >("connectMode" ,0) ),
  //m_authenticationMethod( iConfig.getUntrackedParameter< unsigned int >("authenticationMethod",0) ),
  m_containerName( iConfig.getUntrackedParameter< std::string >("containerName") ),
  m_customMappingFile( iConfig.getUntrackedParameter< std::string >("customMappingFile","") ),
  m_commitInterval( iConfig.getUntrackedParameter< unsigned int >("commitInterval",1) ),
  m_appendIOV( iConfig.getUntrackedParameter< bool >("appendIOV",false) ),
  //m_catalog( iConfig.getUntrackedParameter< std::string >("catalog","") ),
  //m_loadBlobStreamer( iConfig.getUntrackedParameter< bool >("loadBlobStreamer",false) ), 
  //m_messageLevel( iConfig.getUntrackedParameter<unsigned int>("messagelevel",0) ),
  m_loader( new cond::ServiceLoader ),
  m_metadata( new cond::MetaData(m_connect, *m_loader) ),
  m_iov(new cond::IOV),
  m_session( new cond::DBSession(m_connect) ),
  m_payloadWriter(0),
  m_transactionOn(false)
{
 if( m_customMappingFile.empty() ){
   m_payloadWriter=new cond::DBWriter(*m_session,m_containerName);
 }else{
   m_payloadWriter=new cond::DBWriter(*m_session,m_containerName,m_customMappingFile);
 }
 if( m_timetype=="runnumber" ){
   m_endOfTime=(unsigned long long)edm::IOVSyncValue::endOfTime().eventID().run();
 }else{
   m_endOfTime=edm::IOVSyncValue::endOfTime().time().value();
 }
 std::string catalogcontact=iConfig.getUntrackedParameter< std::string >("catalog","");
 m_session->setCatalog(catalogcontact);
 unsigned int authenticationMethod=iConfig.getUntrackedParameter< unsigned int >("authenticationMethod",0);
 bool loadBlobStreamer=iConfig.getUntrackedParameter< bool >("loadBlobStreamer",false);
 unsigned int messageLevel=iConfig.getUntrackedParameter<unsigned int>("messagelevel",0);
 try{
   if( authenticationMethod==1 ){
     m_loader->loadAuthenticationService( cond::XML );
   }else{
     m_loader->loadAuthenticationService( cond::Env );
   }
   if( loadBlobStreamer ){
     m_loader->loadBlobStreamingService();
   }
   switch (messageLevel) {
   case 0 :
     m_loader->loadMessageService(cond::Error);
     break;    
   case 1:
     m_loader->loadMessageService(cond::Warning);
     break;
   case 2:
     m_loader->loadMessageService( cond::Info );
      break;
   case 3:
     m_loader->loadMessageService( cond::Debug );
     break;  
   default:
     m_loader->loadMessageService();
   }
 }catch( const cond::Exception& e){
    throw e;
 }catch( const std::exception& e){
   throw cond::Exception( "PoolDBOutputService::PoolDBOutputService ")<<e.what();
 }catch ( ... ) {
   throw cond::Exception("PoolDBOutputService::PoolDBOutputService unknown error");
 }
 //std::cout<<"module to watch "<< m_clientmodule<<std::endl;
 iAR.watchPostBeginJob(this,&cond::service::PoolDBOutputService::postBeginJob);
 iAR.watchPostEndJob(this,&cond::service::PoolDBOutputService::postEndJob);
 iAR.watchPreModuleConstruction(this,&cond::service::PoolDBOutputService::preModuleConstruction);
 //
 // never get called, useless 
 //
 //iAR.watchPostModuleConstruction(this,&cond::service::PoolDBOutputService::postModuleConstruction);
 //
 // we don't care about sources, useless
 //
 //iAR.watchPreSourceConstruction(this,&cond::service::PoolDBOutputService::preSourceConstruction);
 //iAR.watchPostSourceConstruction(this,&cond::service::PoolDBOutputService::postSourceConstruction);
 iAR.watchPreProcessEvent(this,&cond::service::PoolDBOutputService::preEventProcessing);
 iAR.watchPostProcessEvent(this,&cond::service::PoolDBOutputService::postEventProcessing);
  iAR.watchPreModule(this,&cond::service::PoolDBOutputService::preModule);
  iAR.watchPostModule(this,&cond::service::PoolDBOutputService::postModule);
}
//
// member functions
//

//do connect
void 
cond::service::PoolDBOutputService::postBeginJob()
{
  std::cout<<"PoolDBOutputService: job began."<<std::endl;
  this->connect();
  std::cout<<"PoolDBOutputService: connected "<<std::endl;
}
//do disconnect
void 
cond::service::PoolDBOutputService::postEndJob()
{
  std::cout<<"DBOutput: job about to end"<<std::endl;
  //final commit of payloads in one go if commitInterval underflow
  if( m_transactionOn ){
    m_session->commit();
    m_transactionOn=false;
  }
  if(m_iov->iov.size()!=0){
    if(!m_appendIOV ){
      this->flushIOV();
    }else{
      this->appendIOV();
    }
  }
  this->disconnect();
  std::cout<<"PoolDBOutputService: disconnected "<<std::endl;
}
void 
cond::service::PoolDBOutputService::preModuleConstruction(const edm::ModuleDescription& iDesc)
{
  std::cout<<"DBOutput: preModuleConstruction module label "<<iDesc.moduleLabel_<<std::endl;
  if( m_clientmodule == iDesc.moduleLabel_ ){
    std::cout<<"DBOutput: preModuleConstruction module label "<<iDesc.moduleLabel_<<std::endl;
    return;
  }
  return;
}
//
//use these to control transaction interval
//
void 
cond::service::PoolDBOutputService::preEventProcessing(const edm::EventID& iEvtid, const edm::Timestamp& iTime)
{
  std::cout<<"DBOutput: preEventProcessing Run "<<iEvtid.run()<<" Evt "<<iEvtid.event()<<std::endl;
  if( m_timetype=="runnumber" ){
    m_currentTime=iEvtid.run();
  }else{ //timestamp
    m_currentTime=iTime.value();
  }
  if(!m_transactionOn){
    m_session->startUpdateTransaction();
    m_transactionOn=true;
  }
}
//
//use these to control transaction interval
//
void 
cond::service::PoolDBOutputService::postEventProcessing(const edm::Event& iEvt, const edm::EventSetup& iTime)
{
  std::cout<<"DBOutput: postEventProcessing Run "<<iEvt.id().run()<<" Evt "<<iEvt.id().event()<<std::endl;
  try{
    unsigned int nObjInCache=m_session->DataSvc().cacheSvc().numberOfMarkedObjects();
    std::cout<<"number of Obj in Cache "<<nObjInCache<<std::endl;
    if( m_transactionOn && (nObjInCache>=m_commitInterval) ){
      m_session->commit();
      m_transactionOn=false;
    }
  }catch( const cond::Exception& e){
    throw e;
  }catch(const cms::Exception&e ){
    throw e;
  }catch( const pool::Exception& e){
    throw cond::Exception( "PoolDBOutputService::postEventProcessing ")<<e.what();
  }catch( const std::exception& e){
    throw cond::Exception( "PoolDBOutputService::postEventProcessing ")<<e.what();
  }catch ( ... ) {
    throw cond::Exception("PoolDBOutputService::postEventProcessing unknown error");
  }
}

void 
cond::service::PoolDBOutputService::preModule(const edm::ModuleDescription& iDesc)
{
  if( m_clientmodule==iDesc.moduleLabel_){
    std::cout<<"DBOutput: preModule module label "<<iDesc.moduleLabel_<<std::endl;
    return;
  }
  return;
}

void 
cond::service::PoolDBOutputService::postModule(const edm::ModuleDescription& iDesc)
{
  if( m_clientmodule==iDesc.moduleLabel_ ){
    std::cout<<"DBOutput: postModule module label "<<iDesc.moduleLabel_<<std::endl;
    return;
  }
  return;
}

void
cond::service::PoolDBOutputService::newValidityForOldPayload( const std::string& payloadObjToken, unsigned long long tillTime )
{
  std::cout<<"hello callback "<< payloadObjToken<<" "<<tillTime<<std::endl;
  if(m_appendIOV){
    std::cout<<"PoolDBOutputService::newValidityForOldPayload Error: appending IOV is not allowed"<<std::endl;
    return; //should throw exception here
  }
  m_iov->iov.insert(std::make_pair(tillTime,payloadObjToken));
}
cond::service::PoolDBOutputService::~PoolDBOutputService(){
  std::cout<<" ~PoolDBOutputService "<<std::endl;
  delete m_payloadWriter;
  delete m_metadata;
  delete m_session;
  delete m_loader;
}

void
cond::service::PoolDBOutputService::connect()
{
  try{
    if( m_connectMode==0 ){
      std::cout<<"connecting to db in ReadWriteCreate mode"<<std::endl;
      m_session->connect( cond::ReadWriteCreate );
    }else{
      std::cout<<"connecting to db in ReadWrite mode"<<std::endl;
      m_session->connect( cond::ReadWrite );
    }
    m_metadata->connect();
  }catch( const cond::Exception& e){
    throw e;
  }catch( const std::exception& e){
    throw cond::Exception(std::string("PoolDBOutputService::connect ")+e.what());
  }catch(...) {
    throw cond::Exception(std::string("PoolDBOutputService::connect unknown error") );
  }
}
void 
cond::service::PoolDBOutputService::appendIOV(){
  std::string myoldIOVtoken=m_metadata->getToken(m_tag);
  cond::DBWriter iovwriter(*m_session,"IOV");
  m_session->startUpdateTransaction();
  iovwriter.markDelete<cond::IOV>(myoldIOVtoken);
  std::string iovToken=iovwriter.markWrite<cond::IOV>(m_iov); 
  m_session->commit();
  m_metadata->replaceToken(m_tag, iovToken); 
}
void
cond::service::PoolDBOutputService::flushIOV(){
  m_session->startUpdateTransaction();
  cond::DBWriter iovwriter(*m_session,"IOV");
  m_session->startUpdateTransaction();
  std::string iovToken=iovwriter.markWrite<cond::IOV>(m_iov); 
  m_session->commit();
  m_metadata->addMapping(m_tag, iovToken); 
}
void
cond::service::PoolDBOutputService::disconnect()
{
  m_metadata->disconnect();
  m_session->disconnect();
}
unsigned long long cond::service::PoolDBOutputService::endOfTime(){
  return m_endOfTime;
}
unsigned long long cond::service::PoolDBOutputService::currentTime(){
  return m_currentTime;
}
