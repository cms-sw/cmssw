#include "CondCore/CondDB/interface/Configuration.h"
//
#include "CondCore/DBCommon/interface/CoralServiceManager.h"
#include "CondCore/DBCommon/interface/DbConnectionConfiguration.h"
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

namespace cond {

  namespace persistency {

    SessionConfiguration::SessionConfiguration():
      m_authPath(),
      m_authSys(0),
      m_messageLevel( coral::Error ),
      m_logging( false ),
      m_pluginManager( new cond::CoralServiceManager ){
    }
 
    SessionConfiguration::~SessionConfiguration(){
      delete m_pluginManager;
    }
    
    void SessionConfiguration::setMessageVerbosity( coral::MsgLevel level ){
      m_messageLevel = level;
      m_configured = false;
    }
    
    void SessionConfiguration::setAuthenticationPath( const std::string& p ){
      m_authPath = p;
      m_configured = false;
    }
    
    void SessionConfiguration::setAuthenticationSystem( int authSysCode ){
      m_authSys = authSysCode;
      m_configured = false;
    }
    
    void SessionConfiguration::setLogging( bool flag ){
      m_logging = flag;
      m_configured = false;
    }
    
    void SessionConfiguration::setParameters( const edm::ParameterSet& connectionPset ){
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
      m_configured = false;
    }

    bool SessionConfiguration::isLoggingEnabled() const {
      return m_logging;
    }
    
    void SessionConfiguration::configure( coral::IConnectionServiceConfiguration& coralConfig){
      
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
      m_configured = true;
    }
    
    void SessionConfiguration::configure(  cond::DbConnectionConfiguration& oraConfiguration ) {
      oraConfiguration.setPoolAutomaticCleanUp( false );
      oraConfiguration.setConnectionSharing( false );
      oraConfiguration.setMessageLevel( m_messageLevel );
      oraConfiguration.setAuthenticationPath( m_authPath );
      oraConfiguration.setAuthenticationSystem( m_authSys );
      m_configured = true;
    }
    
    bool SessionConfiguration::isConfigured() const {
      return m_configured;
    }

  }
}
