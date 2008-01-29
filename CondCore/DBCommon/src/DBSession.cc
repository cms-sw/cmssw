#include "CondCore/DBCommon/interface/DBSession.h"
#include <string> 
//#include "CondCore/DBCommon/interface/ConnectMode.h"
//#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "ServiceLoader.h"
#include "SealKernel/Property.h"
#include "SealKernel/PropertyManager.h"
//#include "RelationalAccess/IAuthenticationService.h"
#include <boost/filesystem/operations.hpp>

cond::DBSession::DBSession():m_isActive(false),m_loader(new cond::ServiceLoader),m_connectConfig(new cond::ConnectionConfiguration),m_sessionConfig(new cond::SessionConfiguration),m_usePoolContext(true){ 
}
cond::DBSession::DBSession(bool usePoolContext):m_isActive(false),m_loader(new cond::ServiceLoader),m_connectConfig(new cond::ConnectionConfiguration),m_sessionConfig(new cond::SessionConfiguration),m_usePoolContext(usePoolContext){ 
}
cond::DBSession::~DBSession(){
  delete m_loader;
  delete m_connectConfig;
  delete m_sessionConfig;
}
void cond::DBSession::open(){
  if(m_usePoolContext){
    m_loader->usePOOLContext();
  }else{
    m_loader->useOwnContext();
  }
  m_loader->loadMessageService( m_sessionConfig->messageLevel() );
  m_loader->loadAuthenticationService( m_sessionConfig->authenticationMethod() );
  if( m_sessionConfig->authenticationMethod()==cond::XML ){
    boost::filesystem::path authPath( m_sessionConfig->authName() );
    authPath /= boost::filesystem::path("authentication.xml");
    std::string authName=authPath.string();
    size_t nchildren=m_loader->context()->children();
    for( size_t i=0; i<nchildren; ++i ){
      seal::Handle<seal::PropertyManager> pmgr=m_loader->context()->child(i)->component<seal::PropertyManager>();
      std::string scopeName=pmgr->scopeName();
      //std::cout << "Scope: \"" << scopeName << "\"" << std::endl;
      if( scopeName=="CORAL/Services/XMLAuthenticationService" ){
	pmgr->property("AuthenticationFile")->set(authName);
      }
    }
  }
  m_loader->loadConnectionService( *m_connectConfig );
  //optional
  if(  m_sessionConfig->hasStandaloneRelationalService() ){
    m_loader->loadRelationalService();
  }
  if( m_sessionConfig->hasBlobStreamService() ){
    m_loader->loadBlobStreamingService( m_sessionConfig->blobStreamerName() );
  }
  m_isActive=true;
}
void cond::DBSession::close(){
  m_isActive=false;
}
cond::ServiceLoader& cond::DBSession::serviceLoader(){
  return *m_loader;
}
cond::ConnectionConfiguration& cond::DBSession::connectionConfiguration(){
  return *m_connectConfig;
}
cond::SessionConfiguration& cond::DBSession::sessionConfiguration(){
  return *m_sessionConfig;
}
bool cond::DBSession::isActive() const{
  return m_isActive;
}
void cond::DBSession::purgeConnectionPool(){
  std::vector< seal::IHandle<coral::IConnectionService> > v_svc;
  m_loader->context()->query(v_svc);
  if ( v_svc.empty() ) {
    throw cond::Exception( "Could not locate the connection service" );
  }
  v_svc.front()->purgeConnectionPool();
}
