#include "RecoLuminosity/LumiProducer/interface/DBService.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RelationalAccess/ConnectionService.h"
#include "CoralBase/Exception.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/AccessMode.h"

#include <iostream>
lumi::service::DBService::DBService(const edm::ParameterSet& iConfig,
				    edm::ActivityRegistry& iAR){
  m_svc=new coral::ConnectionService;
  m_dbconfig= new lumi::DBConfig(*m_svc);
  std::string authpath=iConfig.getUntrackedParameter<std::string>("authPath","");
  if( !authpath.empty() ){
    m_dbconfig->setAuthentication(authpath);
  }
}
lumi::service::DBService::~DBService(){
  delete m_dbconfig;
  delete m_svc;
}

void 
lumi::service::DBService::postEndJob(){
}
void 
lumi::service::DBService::preEventProcessing(const edm::EventID& iEvtid, 
					     const edm::Timestamp& iTime){
}
void
lumi::service::DBService::preModule(const edm::ModuleDescription& desc){
}
void 
lumi::service::DBService::preBeginLumi(const edm::LuminosityBlockID& iLumiid,  
				       const edm::Timestamp& iTime ){
}
void
lumi::service::DBService::postModule(const edm::ModuleDescription& desc){
}

coral::ISessionProxy* 
lumi::service::DBService::connectReadOnly( const std::string& connectstring ){
  return m_svc->connect(connectstring, coral::ReadOnly);
}
void
lumi::service::DBService::disconnect( coral::ISessionProxy* session ){
  delete session;
}
lumi::DBConfig&
lumi::service::DBService::DBConfig(){
  return *m_dbconfig;
}
void
lumi::service::DBService::setupWebCache(){
}
