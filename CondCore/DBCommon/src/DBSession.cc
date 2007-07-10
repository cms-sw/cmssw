//seal includes
#include "PluginManager/PluginManager.h"
#include "SealKernel/IMessageService.h"
#include "SealKernel/Property.h"
#include "SealKernel/PropertyManager.h"
//coral includes
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/IRelationalService.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IMonitoringService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
//pool includes
#include "POOLCore/POOLContext.h"
#include "RelationalStorageService/IBlobStreamingService.h"
//local includes
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
// pool includes
#include "POOLCore/POOLContext.h"
#include <boost/filesystem/operations.hpp>
cond::DBSession::DBSession(){ 
  seal::PluginManager* pm = seal::PluginManager::get();
  if ( ! pm ) {
    throw cond::Exception( "Could not get the plugin manager instance" );
  }
  pm->initialise();
  m_context=pool::POOLContext::context();
  m_loader = new seal::ComponentLoader( m_context.get() );
  m_sessionConfig = new cond::SessionConfiguration;
}
cond::DBSession::~DBSession(){
  delete m_sessionConfig;
}
void cond::DBSession::open(){
  m_loader->load("SEAL/Services/MessageService");
  std::vector< seal::IHandle<seal::IMessageService> > v_msgSvc;
  m_context->query( v_msgSvc );
  if ( v_msgSvc.empty() ) {
    throw cond::Exception( "could not locate the seal message service" );
  }
  switch ( m_sessionConfig->messageLevel() ) {
  case cond::Error :
    v_msgSvc.front()->setOutputLevel( seal::Msg::Error );
    break;
  case cond::Warning :
    v_msgSvc.front()->setOutputLevel( seal::Msg::Warning );
    break;
  case cond::Debug :
    v_msgSvc.front()->setOutputLevel( seal::Msg::Debug );
    break;
  case cond::Info :
    v_msgSvc.front()->setOutputLevel( seal::Msg::Info );
    break;
  default:
    v_msgSvc.front()->setOutputLevel( seal::Msg::Error );
  } 
  //load authentication service
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
  std::vector< seal::IHandle<coral::IAuthenticationService> > v_authsvc;
  if( m_sessionConfig->authenticationMethod()== cond::XML ) {
    m_loader->load( "CORAL/Services/XMLAuthenticationService" );
  }else{
    m_loader->load( "CORAL/Services/EnvironmentAuthenticationService" );
  }
  m_context->query(v_authsvc);
  if ( v_authsvc.empty() ) {
    throw cond::Exception( "Could not locate authentication service" );
  }
  //load relational service
  m_loader->load( "CORAL/Services/RelationalService" );
  //load connection service
  m_loader->load( "CORAL/Services/ConnectionService" );
  
  m_con=m_context->query<coral::IConnectionService>( "CORAL/Services/ConnectionService" ).get();
  if (! m_con ) {
    throw cond::Exception( "could not locate the coral connection service" );
  }
  coral::IConnectionServiceConfiguration& conserviceConfig = connectionService().configuration();
  cond::ConnectionConfiguration* conConfig=m_sessionConfig->connectionConfiguration();
  if( conConfig ){
  if( conConfig->isConnectionSharingEnabled() ){
    conserviceConfig.enableConnectionSharing();
  }
  conserviceConfig.setConnectionRetrialPeriod( conConfig->connectionRetrialPeriod() );
  conserviceConfig.setConnectionRetrialTimeOut( conConfig->connectionRetrialTimeOut() );
  conserviceConfig.setConnectionTimeOut( conConfig->connectionTimeOut() );
  conserviceConfig.setMonitoringLevel( conConfig->monitorLevel() ); 
  if( m_sessionConfig->hasBlobStreamService() ){
    std::string streamerName=m_sessionConfig->blobStreamerName();
      if(streamerName.empty()){
	m_loader->load( "COND/Services/DefaultBlobStreamingService" );
      }else{
	m_loader->load(streamerName);
      }
    std::vector< seal::IHandle<pool::IBlobStreamingService> > v_blobsvc;
    m_context->query( v_blobsvc );
    if ( v_blobsvc.empty() ) {
      throw cond::Exception( "could not locate the BlobStreamingService" );
    }
  }
  }
}
coral::IConnectionService& 
cond::DBSession::connectionService(){
  return *m_con;
}
coral::IRelationalService& 
cond::DBSession::relationalService(){
  std::vector< seal::IHandle<coral::IRelationalService> > v_svc;
  m_context->query( v_svc );
  if ( v_svc.empty() ) {
    throw cond::Exception( "Could not locate the relational service" );
  }
  return *(v_svc.front().get());
}
coral::IAuthenticationService& 
cond::DBSession::authenticationService() const{
   std::vector< seal::IHandle<coral::IAuthenticationService> > v_svc;
   m_context->query( v_svc );
   if ( v_svc.empty() ) {
     throw cond::Exception( "Could not locate the authentication service" );
   }
   return *(v_svc.front().get());
}
const coral::IMonitoringReporter& 
cond::DBSession::monitoringReporter() const{
  return m_con->monitoringReporter();
}
coral::IWebCacheControl& 
cond::DBSession::webCacheControl(){
  return m_con->webCacheControl();
}
pool::IBlobStreamingService& 
cond::DBSession::blobStreamingService(){
  std::vector< seal::IHandle<pool::IBlobStreamingService> > v_svc;
  m_context->query( v_svc );
  if ( v_svc.empty() ) {
    throw cond::Exception( "Could not locate the BlobStreamingService" );
  }
  return *(v_svc.front().get());
}
cond::SessionConfiguration& 
cond::DBSession::configuration(){
  return *m_sessionConfig;
}
