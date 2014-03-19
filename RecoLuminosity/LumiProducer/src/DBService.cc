#include "RecoLuminosity/LumiProducer/interface/DBService.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RelationalAccess/ConnectionService.h"
#include "CoralBase/Exception.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/AccessMode.h"

#include <iostream>
lumi::service::DBService::DBService(const edm::ParameterSet& iConfig){
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

coral::ISessionProxy* 
lumi::service::DBService::connectReadOnly( const std::string& connectstring ){
  return m_svc->connect(connectstring, coral::ReadOnly);
}
void
lumi::service::DBService::disconnect( coral::ISessionProxy* session ){
  delete session;
}

