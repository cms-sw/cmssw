//coral includes
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/IRelationalService.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
//pool includes
#include "POOLCore/IBlobStreamingService.h"
//local includes
#include "CondCore/DBCommon/interface/DBSession.h"

#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"

#include "CondCore/DBCommon/interface/Exception.h"
// pool includes
#include <boost/filesystem/operations.hpp>
//#include <iostream>
cond::DBSession::DBSession(ConfDefaults confDef){ 
  config(confDef);
}

void cond::DBSession::config(ConfDefaults confDef) {
  if (confDef!=coralDefaults) {
    configuration().connectionConfiguration()->disablePoolAutomaticCleanUp();
    configuration().connectionConfiguration()->setConnectionTimeOut(0);
  }

}


cond::DBSession::~DBSession(){}

void cond::DBSession::open(){
  switch ( m_sessionConfig.messageLevel() ) {
  case cond::Error :
    { coral::MessageStream::setMsgVerbosity( coral::Error );
      break;
    }
  case cond::Warning :
    { coral::MessageStream::setMsgVerbosity( coral::Warning );
      break;
    }
  case cond::Debug :
    { coral::MessageStream::setMsgVerbosity( coral::Debug );
      break;
    }
  case cond::Info :
    { coral::MessageStream::setMsgVerbosity( coral::Info );
      break;
    }
  default:
    { coral::MessageStream::setMsgVerbosity( coral::Error ); }
  }
  //load authentication service
  if( m_sessionConfig.authenticationMethod()== cond::XML ) {
    
    boost::filesystem::path authPath( m_sessionConfig.authName() );
    if(boost::filesystem::is_directory(m_sessionConfig.authName())){
      authPath /= boost::filesystem::path("authentication.xml");
    }
    std::string authName=authPath.string();
    coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(authName);
    coral::Context::instance().loadComponent( "COND/Services/XMLAuthenticationService",&m_pluginmanager);
  }else{
    coral::Context::instance().loadComponent( "CORAL/Services/EnvironmentAuthenticationService");
  }
  coral::Context::instance().loadComponent( "CORAL/Services/ConnectionService");
  coral::IAuthenticationService& authServ = authenticationService();
  coral::IConnectionService& connSrv = connectionService();
  connSrv.configuration().setAuthenticationService(authServ);
  coral::IConnectionServiceConfiguration& conserviceConfig = connectionService().configuration();
  cond::ConnectionConfiguration* conConfig=m_sessionConfig.connectionConfiguration();
  if(m_sessionConfig.isSQLMonitoringOn()){
    coral::Context::instance().loadComponent( "COND/Services/SQLMonitoringService",&m_pluginmanager);
    conConfig->setMonitorLevel(coral::monitor::Trace);
  }
  if( conConfig ){
    if( conConfig->isConnectionSharingEnabled() ){
      conserviceConfig.enableConnectionSharing();
    }
    if( conConfig->isPoolAutomaticCleanUpEnabled() ){
      conserviceConfig.enablePoolAutomaticCleanUp();
    }else{
      conserviceConfig.disablePoolAutomaticCleanUp();
    }
    conserviceConfig.setConnectionRetrialPeriod( conConfig->connectionRetrialPeriod() );
    conserviceConfig.setConnectionRetrialTimeOut( conConfig->connectionRetrialTimeOut() );
    conserviceConfig.setConnectionTimeOut( conConfig->connectionTimeOut() );
    conserviceConfig.setMonitoringLevel( conConfig->monitorLevel() );

    if( m_sessionConfig.hasBlobStreamService() ){
      std::string streamerName=m_sessionConfig.blobStreamerName();
      if(streamerName.empty()){
	coral::Context::instance().loadComponent( "COND/Services/TBufferBlobStreamingService", &m_pluginmanager );
      }else{
	coral::Context::instance().loadComponent(streamerName, &m_pluginmanager);
      }
    }
  }
}
coral::IConnectionService& 
cond::DBSession::connectionService(){
  return *(coral::Context::instance().query<coral::IConnectionService>());
}
coral::IRelationalService& 
cond::DBSession::relationalService(){
  return *(coral::Context::instance().query<coral::IRelationalService>());
}
coral::IAuthenticationService& 
cond::DBSession::authenticationService(){
  return *(coral::Context::instance().query<coral::IAuthenticationService>());
}
const coral::IMonitoringReporter& 
cond::DBSession::monitoringReporter() const{
  return coral::Context::instance().query<coral::IConnectionService>()->monitoringReporter();
}
coral::IWebCacheControl& 
cond::DBSession::webCacheControl(){
  return coral::Context::instance().query<coral::IConnectionService>()->webCacheControl();
}
pool::IBlobStreamingService& 
cond::DBSession::blobStreamingService(){
  return *(coral::Context::instance().query<pool::IBlobStreamingService>());
}
cond::SessionConfiguration& 
cond::DBSession::configuration(){
  return m_sessionConfig;
}
