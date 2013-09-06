#include "CondCore/CondDB/interface/Configuration.h"
//
#include "CondCore/DBCommon/interface/CoralServiceManager.h"
#include "CondCore/DBCommon/interface/Auth.h"
// CMSSW includes
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
// coral includes
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
// externals
#include <boost/filesystem/operations.hpp>

conddb::Configuration::Configuration():
  m_authPath(),
  m_authSys(0),
  m_messageLevel( coral::Error ),
  m_logging( false ),
  m_pluginManager( new cond::CoralServiceManager ){
}
 
conddb::Configuration::~Configuration(){
  delete m_pluginManager;
}

void conddb::Configuration::setMessageVerbosity( coral::MsgLevel level ){
  m_messageLevel = level;
}
    
void conddb::Configuration::setAuthenticationPath( const std::string& p ){
  m_authPath = p;
}
    
void conddb::Configuration::setAuthenticationSystem( int authSysCode ){
  m_authSys = authSysCode;
}
    
void conddb::Configuration::setLogging( bool flag ){
  m_logging = flag;
}
    
void conddb::Configuration::setParameters( const edm::ParameterSet& connectionPset ){
  setAuthenticationPath( connectionPset.getUntrackedParameter<std::string>("authenticationPath","") );
  setAuthenticationSystem( connectionPset.getUntrackedParameter<int>("authenticationSystem",0) );
  int messageLevel = connectionPset.getUntrackedParameter<int>("messageLevel",0);
  coral::MsgLevel level = coral::Error;
  switch (messageLevel) {
    case 0 :
      level = coral::Error;
      break;    
    case 1:
      level = coral::Warning;
      break;
    case 2:
      level = coral::Info;
      break;
    case 3:
      level = coral::Debug;
      break;
    default:
      level = coral::Error;
  }
  setMessageVerbosity(level);
  setLogging( connectionPset.getUntrackedParameter<bool>("logging",false) );
}

bool conddb::Configuration::isLoggingEnabled() const {
  return m_logging;
}
    
void conddb::Configuration::configure( coral::IConnectionServiceConfiguration& coralConfig) const {

  coralConfig.disablePoolAutomaticCleanUp();
  coralConfig.disableConnectionSharing();
 // message streaming
  coral::MessageStream::setMsgVerbosity( m_messageLevel );
  std::string authServiceName("CORAL/Services/EnvironmentAuthenticationService");
  std::string authPath = m_authPath;
  // authentication
  if( authPath.empty() ){
    // first try to check the env...
    const char* authEnv = ::getenv( cond::Auth::COND_AUTH_PATH );
    if(authEnv){
      authPath += authEnv;
    } 
  }
  int authSys = m_authSys;
  // first attempt, look at the env...
  const char* authSysEnv = ::getenv( cond::Auth::COND_AUTH_SYS );
  if( authSysEnv ){
    authSys = ::atoi( authSysEnv );
  }
  if( authSys !=CondDbKey && authSys != CoralXMLFile ){
    // take the default
    authSys = CondDbKey;
  }  
  std::string servName("");
  if( authSys == CondDbKey ){
    if( authPath.empty() ){
      const char* authEnv = ::getenv("HOME");
      if(authEnv){
	authPath += authEnv;
      } 
    }
    servName = "COND/Services/RelationalAuthenticationService";     
    //edm::LogInfo("DbSessionInfo") << "Authentication using Keys";  
  } else if( authSys == CoralXMLFile ){
    if( authPath.empty() ){
      authPath = ".";
    }
    servName = "COND/Services/XMLAuthenticationService";  
    //edm::LogInfo("DbSessionInfo") << "Authentication using XML File";  
  }
  if( !authPath.empty() ){
    authServiceName = servName;    
    coral::Context::instance().PropertyManager().property(cond::Auth::COND_AUTH_PATH_PROPERTY)->set(authPath);  
    coral::Context::instance().loadComponent( authServiceName, m_pluginManager );
  }
  coralConfig.setAuthenticationService( authServiceName );
}
