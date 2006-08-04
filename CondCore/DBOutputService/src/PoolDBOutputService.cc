#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataSvc/RefException.h"

#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "serviceCallbackToken.h"
#include <iostream>
#include <vector>
cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ): 
  m_connect( iConfig.getParameter< std::string > ("connect") ),
  m_timetype( iConfig.getParameter< std::string >("timetype") ),
  m_connectMode( iConfig.getUntrackedParameter< unsigned int >("connectMode" ,0) ),
  //m_customMappingFile( iConfig.getUntrackedParameter< std::string >("customMappingFile","") ),
  m_loader( new cond::ServiceLoader ),
  m_metadata( 0 ),
  m_session( 0 ),
  m_iovWriter( 0 ),
  m_dbstarted( false )
{
  if( m_timetype=="runnumber" ){
    m_endOfTime=(unsigned long long)edm::IOVSyncValue::endOfTime().eventID().run();
  }else{
    m_endOfTime=edm::IOVSyncValue::endOfTime().time().value();
  }
  std::string catalogcontact=iConfig.getUntrackedParameter< std::string >("catalog","");
  bool loadBlobStreamer=iConfig.getUntrackedParameter< bool >("loadBlobStreamer",false);
  unsigned int authenticationMethod=iConfig.getUntrackedParameter< unsigned int >("authenticationMethod",0);
  unsigned int messageLevel=iConfig.getUntrackedParameter<unsigned int>("messagelevel",0);
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toPut=iConfig.getParameter<Parameters>("toPut");
  for(Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut) {
    cond::service::serviceCallbackRecord thisrecord;
    thisrecord.m_containerName = itToPut->getUntrackedParameter<std::string>("containerName");
    thisrecord.m_appendIOV = itToPut->getUntrackedParameter<bool>("appendIOV",false);
    thisrecord.m_tag = itToPut->getParameter<std::string>("tag");
    m_callbacks.insert(std::make_pair(cond::service::serviceCallbackToken::build(thisrecord.m_containerName),thisrecord));
  }
  try{
    if( authenticationMethod==1 ){
      m_loader->loadAuthenticationService( cond::XML );
    }else{
      m_loader->loadAuthenticationService( cond::Env );
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
    if( loadBlobStreamer ){
      m_loader->loadBlobStreamingService();
    }
  }catch( const cond::Exception& e){
    throw e;
  }catch( const std::exception& e){
    throw cond::Exception( "PoolDBOutputService::PoolDBOutputService ")<<e.what();
  }catch ( ... ) {
    throw cond::Exception("PoolDBOutputService::PoolDBOutputService unknown error");
  }
  iAR.watchPreProcessEvent(this,&cond::service::PoolDBOutputService::preEventProcessing);
  //iAR.watchPostBeginJob(this,&cond::service::PoolDBOutputService::postBeginJob);
  iAR.watchPostEndJob(this,&cond::service::PoolDBOutputService::postEndJob);
  m_metadata=new cond::MetaData(m_connect, *m_loader);
  m_session=new cond::DBSession(m_connect);
  m_session->setCatalog(catalogcontact);
  m_iovWriter=new cond::DBWriter(*m_session,"cond::IOV");  
}
void 
cond::service::PoolDBOutputService::initDB()
{
  if(m_dbstarted) return;
  try{
    this->connect();
    m_session->startUpdateTransaction();
    std::map<size_t, cond::service::serviceCallbackRecord>::iterator it;
    for(it=m_callbacks.begin();it!=m_callbacks.end(); ++it){
      cond::service::serviceCallbackRecord& myrecord=it->second;
      if( myrecord.m_appendIOV ){
	myrecord.m_iovToken=m_metadata->getToken(myrecord.m_tag);
	myrecord.m_iov=m_iovWriter->markUpdate<cond::IOV>(myrecord.m_iovToken);
      }else{
	myrecord.m_iov=new cond::IOV;
	myrecord.m_iovToken=m_iovWriter->markWrite<cond::IOV>(myrecord.m_iov);
      }
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
  m_dbstarted=true;
  //std::cout<<"PoolDBOutputService: connected "<<std::endl;
}
void 
cond::service::PoolDBOutputService::postEndJob()
{
  if(!m_dbstarted) return;
  m_session->commit();
  std::map<size_t, cond::service::serviceCallbackRecord>::iterator it;
  for(it=m_callbacks.begin(); it!=m_callbacks.end(); ++it){
    cond::service::serviceCallbackRecord& myrecord=it->second;
    if(!myrecord.m_appendIOV){
      m_metadata->addMapping(myrecord.m_tag, myrecord.m_iovToken); 
    }
  }
  this->disconnect();
}
void 
cond::service::PoolDBOutputService::preEventProcessing(const edm::EventID& iEvtid, const edm::Timestamp& iTime)
{
  if( m_timetype=="runnumber" ){
    m_currentTime=iEvtid.run();
  }else{ //timestamp
    m_currentTime=iTime.value();
  }
}
void
cond::service::PoolDBOutputService::newValidityForOldPayload( const std::string& payloadObjToken, unsigned long long tillTime, size_t callbackToken )
{
  std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.find(callbackToken);
  if(it==m_callbacks.end()) throw cond::Exception(std::string("PoolDBOutputService::newValidityForOldPayload: unregistered callback token"));
  cond::service::serviceCallbackRecord& myrecord=it->second;
  if(myrecord.m_appendIOV){
    throw cond::Exception("PoolDBOutputService::newValidityForOldPayload: appending IOV is not allowed");
  }
  myrecord.m_iov->iov.insert(std::make_pair(tillTime,payloadObjToken));
}
cond::service::PoolDBOutputService::~PoolDBOutputService(){
  m_callbacks.clear();
  delete m_iovWriter;
  delete m_session;
  delete m_metadata;
  delete m_loader;  
}
size_t cond::service::PoolDBOutputService::callbackToken(const std::string& containerName) const {
  return cond::service::serviceCallbackToken::build(containerName);
}
void
cond::service::PoolDBOutputService::connect()
{
  try{
    if( m_connectMode==0 ){
      //std::cout<<"PoolDBOutputService::connect ReadWriteCreate"<<std::endl;
      m_session->connect( cond::ReadWriteCreate );
      m_metadata->connect(cond::ReadWriteCreate);
    }else{
      //std::cout<<"metadata connection ReadWrite"<<std::endl;
      m_session->connect( cond::ReadWrite );
      m_metadata->connect(cond::ReadWrite);
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
  m_metadata->disconnect();
  m_session->disconnect();
}
unsigned long long cond::service::PoolDBOutputService::endOfTime() const{
  return m_endOfTime;
}
unsigned long long cond::service::PoolDBOutputService::currentTime() const{
  return m_currentTime;
}
