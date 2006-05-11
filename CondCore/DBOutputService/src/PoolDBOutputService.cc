#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataSvc/IDataSvc.h"
#include "DataSvc/ICacheSvc.h"
#include "DataSvc/RefException.h"

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
  m_iov( 0 ),
  m_session( new cond::DBSession(m_connect) ),
  m_payloadWriter( 0 ),
  m_iovWriter( 0 )
  //m_transactionOn(false)
{
  //std::cout<<"PoolDBOutputService"<<std::endl;
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
  m_iovWriter=new cond::DBWriter(*m_session,"IOV");
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
    if(m_appendIOV){
      m_metadata->connect(cond::ReadOnly);
      m_iovToken=m_metadata->getToken(m_tag);
      m_metadata->disconnect();
    }else{
      m_iov=new cond::IOV;
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
  //iAR.watchPreModuleConstruction(this,&cond::service::PoolDBOutputService::preModuleConstruction);
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
  //iAR.watchPostProcessEvent(this,&cond::service::PoolDBOutputService::postEventProcessing);
  //iAR.watchPreModule(this,&cond::service::PoolDBOutputService::preModule);
  //iAR.watchPostModule(this,&cond::service::PoolDBOutputService::postModule);
}
//
// member functions
//

//do connect
void 
cond::service::PoolDBOutputService::postBeginJob()
{
  //std::cout<<"PoolDBOutputService::postBeginJob"<<std::endl;
  try{
    //std::cout<<"Pool connect "<<std::endl;
    this->connect();
    //std::cout<<"start pool update transaction "<<std::endl;
    m_session->startUpdateTransaction();
    if( m_appendIOV ){
      m_iov=m_iovWriter->markUpdate<cond::IOV>(m_iovToken);
    }
  }catch( const pool::RefException& er){
    //std::cerr<<"caught RefException "<<er.what()<<std::endl;
    throw cms::Exception( er.what() );
  }catch( const pool::Exception& er ){
    //std::cerr<<"caught pool Exception "<<er.what()<<std::endl;
    throw cms::Exception( er.what() );
  }catch( const std::exception& er ){
    throw cms::Exception( er.what() );
  }catch(...){
    throw cms::Exception( "Funny error" );
  }
  //std::cout<<"PoolDBOutputService: connected "<<std::endl;
}
//do disconnect
void 
cond::service::PoolDBOutputService::postEndJob()
{
  //std::cout<<"PoolDBOutputService::postEndJob"<<std::endl;
  //final commit of payloads in one go if commitInterval underflow
  //if( m_transactionOn ){
  // std::cout<<"about to do final commit "<<std::endl;
  //m_session->commit();
  //std::cout<<"committed "<<std::endl;
  //m_transactionOn=false;
  // }
  if(!m_appendIOV ){
    if(m_iov->iov.size()!=0){
      m_iovToken=m_iovWriter->markWrite<cond::IOV>(m_iov); 
    }
  }  
  //std::cout<<"Pool commit"<<std::endl;
  m_session->commit();
  //std::cout<<"Pool disconnect"<<std::endl;
  this->disconnect();
  if(!m_appendIOV){
    if( m_connectMode==0 ){
      //std::cout<<"metadata connection ReadWriteCreate"<<std::endl;
      m_metadata->connect(cond::ReadWriteCreate);
    }else{
      //std::cout<<"metadata connection ReadWrite"<<std::endl;
      m_session->connect( cond::ReadWrite );
    }
    m_metadata->addMapping(m_tag, m_iovToken); 
    //std::cout<<"metadata disconnect"<<std::endl;
    m_metadata->disconnect();
  }
}

//
//use these to control transaction interval
//

void 
cond::service::PoolDBOutputService::preEventProcessing(const edm::EventID& iEvtid, const edm::Timestamp& iTime)
{
  if( m_timetype=="runnumber" ){
    m_currentTime=iEvtid.run();
  }else{ //timestamp
    m_currentTime=iTime.value();
  }
}
//
//use these to control transaction intervals
//
void 
cond::service::PoolDBOutputService::postEventProcessing(const edm::Event& iEvt, const edm::EventSetup& iTime)
{
}

void
cond::service::PoolDBOutputService::newValidityForOldPayload( const std::string& payloadObjToken, unsigned long long tillTime )
{
  if(m_appendIOV){
    throw cond::Exception("PoolDBOutputService::newValidityForOldPayload: appending IOV is not allowed");
  }
  m_iov->iov.insert(std::make_pair(tillTime,payloadObjToken));
}
cond::service::PoolDBOutputService::~PoolDBOutputService(){
  delete m_payloadWriter;
  delete m_iovWriter;
  delete m_metadata;
  delete m_session;
  delete m_loader;
}

void
cond::service::PoolDBOutputService::connect()
{
  try{
    if( m_connectMode==0 ){
      //std::cout<<"PoolDBOutputService::connect ReadWriteCreate"<<std::endl;
      m_session->connect( cond::ReadWriteCreate );
    }else{
      //std::cout<<"PoolDBOutputService::connect ReadWrite"<<std::endl;
      m_session->connect( cond::ReadWrite );
    }
  }catch( const cond::Exception& e){
    throw e;
  }catch( const std::exception& e){
    throw cond::Exception(std::string("PoolDBOutputService::connect ")+e.what());
  }catch(...) {
    throw cond::Exception(std::string("PoolDBOutputService::connect unknown error") );
  }
}

void
cond::service::PoolDBOutputService::disconnect()
{
  //std::cout<<"PoolDBOutputService::disconnect"<<std::endl;
  m_session->disconnect();
}
unsigned long long cond::service::PoolDBOutputService::endOfTime(){
  return m_endOfTime;
}
unsigned long long cond::service::PoolDBOutputService::currentTime(){
  return m_currentTime;
}
