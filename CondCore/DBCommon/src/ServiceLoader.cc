#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "PluginManager/PluginManager.h"
#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"
#include "SealKernel/Component.h"
#include "SealKernel/IMessageService.h"
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/IRelationalService.h"
#include "CondCore/BlobStreamingService/interface/BlobStreamingService.h"
#include "POOLCore/POOLContext.h"
cond::ServiceLoader::ServiceLoader(){
  m_context=new seal::Context();
  seal::PluginManager* pm = seal::PluginManager::get();
  pm->initialise();
  m_loader = new seal::ComponentLoader( m_context );
}
cond::ServiceLoader::~ServiceLoader(){
  delete m_context;
}
seal::IMessageService& cond::ServiceLoader::loadMessageService( cond::MessageLevel level ){
  pool::POOLContext::loadComponent( "SEAL/Services/MessageService" );
  m_loader->load("SEAL/Services/MessageService");
  std::vector< seal::IHandle<seal::IMessageService> > v_msgSvc;
  m_context->query( v_msgSvc );
  if ( v_msgSvc.empty() ) {
    throw cond::Exception( "could not locate the seal message service" );
  }
  switch ( level ) {
  case cond::Error :
    v_msgSvc.front()->setOutputLevel( seal::Msg::Error );
    pool::POOLContext::setMessageVerbosityLevel( seal::Msg::Error );
    break;
  case cond::Warning :
    v_msgSvc.front()->setOutputLevel( seal::Msg::Warning );
    pool::POOLContext::setMessageVerbosityLevel( seal::Msg::Warning );
    break;
  case cond::Debug :
    v_msgSvc.front()->setOutputLevel( seal::Msg::Debug );
    pool::POOLContext::setMessageVerbosityLevel( seal::Msg::Debug );
    break;
  case cond::Info :
    v_msgSvc.front()->setOutputLevel( seal::Msg::Info );
    pool::POOLContext::setMessageVerbosityLevel( seal::Msg::Info );
    break;
  default:
    v_msgSvc.front()->setOutputLevel( seal::Msg::Error );
    pool::POOLContext::setMessageVerbosityLevel( seal::Msg::Error );
  } 
  return *(v_msgSvc.front());
}
bool cond::ServiceLoader::hasMessageService() const{
  std::vector< seal::IHandle<seal::IMessageService> > v_msgSvc;
  m_context->query( v_msgSvc );
  if( !v_msgSvc.empty() ) return true;
  return false;
}
coral::IAuthenticationService& cond::ServiceLoader::loadAuthenticationService( cond::AuthenticationMethod method){
  std::vector< seal::IHandle<coral::IAuthenticationService> > v_svc;
  switch ( method ) {
  case cond::Env :
    m_loader->load( "CORAL/Services/EnvironmentAuthenticationService" );
    m_context->query( v_svc );
    if ( v_svc.empty() ) {
      throw cond::Exception( "could not locate the coral authentication service" );
    }
    break;
  case cond::XML :
    m_loader->load( "CORAL/Services/XMLAuthenticationService" );
    m_context->query( v_svc );
    if ( v_svc.empty() ) {
      throw cond::Exception( "could not locate the coral authentication service" );
    }
    break;
  default:
    m_loader->load( "CORAL/Services/EnvironmentAuthenticationService" );
    m_context->query( v_svc );
    if ( v_svc.empty() ) {
      throw cond::Exception( "could not locate the coral authentication service" );
    }
  }
  return *(v_svc.front());
}
bool cond::ServiceLoader::hasAuthenticationService() const{
  std::vector< seal::IHandle<coral::IAuthenticationService> > v_svc;
  m_context->query( v_svc );
  if( !v_svc.empty() ) return true;
  return false;
}
coral::IRelationalService& cond::ServiceLoader::loadRelationalService(){
  m_loader->load( "CORAL/Services/RelationalService" );
  std::vector< seal::IHandle<coral::IRelationalService> > v_svc;
  m_context->query( v_svc );
  if ( v_svc.empty() ) {
    throw cond::Exception( "could not locate the coral relational service" );
  }
  return *(v_svc.front());
}
void cond::ServiceLoader::loadConnectionService(){
}
pool::IBlobStreamingService& cond::ServiceLoader::loadBlobStreamingService(){
  m_loader->load( "COND/Services/DefaultBlobStreamingService" );
  std::vector< seal::IHandle<pool::IBlobStreamingService> > v_svc;
  m_context->query( v_svc );
  if ( v_svc.empty() ) {
    throw cond::Exception( "could not locate the BlobStreamingService" );
  }
  return *(v_svc.front());
}
pool::IBlobStreamingService& cond::ServiceLoader::loadBlobStreamingService( const std::string& componentName ){
  m_loader->load( componentName );
  std::vector< seal::IHandle<pool::IBlobStreamingService> > v_svc;
  m_context->query( v_svc );
  if ( v_svc.empty() ) {
    throw cond::Exception( std::string("could not locate the BlobStreamingService ")+componentName );
  }
  return *(v_svc.front());
}

