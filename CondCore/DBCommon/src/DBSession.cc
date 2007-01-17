#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "ServiceLoader.h"
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
  //necessary
  m_loader->loadMessageService( m_sessionConfig->messageLevel() );
  m_loader->loadConnectionService( *m_connectConfig );
  m_loader->loadAuthenticationService( m_sessionConfig->authenticationMethod() );
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
