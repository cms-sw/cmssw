#include "PluginManager/PluginManager.h"
//#include "SealKernel/ComponentLoader.h"
#include "SealKernel/IMessageService.h"
#include "SealKernel/Property.h"
#include "SealKernel/PropertyManager.h"
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/IRelationalService.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IMonitoringService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "POOLCore/POOLContext.h"
#include "RelationalStorageService/IBlobStreamingService.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include <boost/filesystem/operations.hpp>
cond::DBSession::DBSession(){ 
  seal::PluginManager* pm = seal::PluginManager::get();
  if ( ! pm ) {
    throw cond::Exception( "Could not get the plugin manager instance" );
  }
  pm->initialise();
  if(!m_context){
    m_context=pool::POOLContext::context();
    m_loader = new seal::ComponentLoader( m_context.get() );
  }
}
cond::DBSession::~DBSession(){
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
  //load monitoring service
  m_loader->load( "CORAL/Services/MonitoringService" );
  //load connection service
  m_loader->load( "CORAL/Services/ConnectionService" );
  
  m_con=m_context->query<coral::IConnectionService>( "CORAL/Services/ConnectionService" ).get();
  if (! m_con ) {
    throw cond::Exception( "could not locate the coral connection service" );
  }
  /*coral::IConnectionServiceConfiguration& conserviceConfig = iHandle->configuration();
  if( config.isConnectionSharingEnabled() ){
    conserviceConfig.enableConnectionSharing();
  }
  conserviceConfig.setConnectionRetrialPeriod( config.connectionRetrialPeriod() );
  conserviceConfig.setConnectionRetrialTimeOut( config.connectionRetrialTimeOut() );
  conserviceConfig.setConnectionTimeOut( config.connectionTimeOut() );
  */
  /*if( m_sessionConfig->loadBlobStreamer() ){
    std::string streamerName=m_sessionConfig->blobStreamerName()
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
  */
}
