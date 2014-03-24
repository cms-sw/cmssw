#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "DbConnectionString.h"
//
#include "CondCore/DBCommon/interface/CoralServiceManager.h"
#include "CondCore/DBCommon/interface/Auth.h"
// CMSSW includes
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// coral includes
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/IWebCacheControl.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
// externals
#include <boost/filesystem/operations.hpp>

namespace cond {

  namespace persistency {
   
    ConnectionPool::ConnectionPool():
      m_authPath(),
      m_authSys(0),
      m_messageLevel( coral::Error ),
      m_loggingEnabled( false ),
      m_pluginManager( new cond::CoralServiceManager ),
      m_refreshtablelist(){
      m_refreshtablelist.reserve(6);
      //table names for IOVSequence in the old POOL mapping
      m_refreshtablelist.push_back("IOV");
      m_refreshtablelist.push_back("IOV_DATA");
      //table names for IOVSequence in ORA
      m_refreshtablelist.push_back("ORA_C_COND_IOVSEQUENCE");
      m_refreshtablelist.push_back("ORA_C_COND_IOVSEQU_A0");
      m_refreshtablelist.push_back("ORA_C_COND_IOVSEQU_A1");
      //table names for IOVSequence in CONDDB
      m_refreshtablelist.push_back("TAG");
      configure();
    }
 
    ConnectionPool::~ConnectionPool(){
      delete m_pluginManager;
    }
    
    void ConnectionPool::setAuthenticationPath( const std::string& p ){
      m_authPath = p;
    }
    
    void ConnectionPool::setAuthenticationSystem( int authSysCode ){
      m_authSys = authSysCode;
    }
    
    void ConnectionPool::setLogging( bool flag ){
      m_loggingEnabled = flag;
    }
    
    void ConnectionPool::setParameters( const edm::ParameterSet& connectionPset ){
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

    bool ConnectionPool::isLoggingEnabled() const {
      return m_loggingEnabled;
    }
    
    void ConnectionPool::configure( coral::IConnectionServiceConfiguration& coralConfig){
      
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
      } else if( authSys == CoralXMLFile ){
	if( authPath.empty() ){
	  authPath = ".";
	}
	servName = "COND/Services/XMLAuthenticationService";  
      }
      if( !authPath.empty() ){
	authServiceName = servName;    
	coral::Context::instance().PropertyManager().property(cond::Auth::COND_AUTH_PATH_PROPERTY)->set(authPath);  
	coral::Context::instance().loadComponent( authServiceName, m_pluginManager );
      }
      coralConfig.setAuthenticationService( authServiceName );
    }
    
    void ConnectionPool::configure() {
      coral::ConnectionService connServ;
      configure( connServ.configuration() );
    }

    Session ConnectionPool::createSession( const std::string& connectionString, const std::string& transactionId, bool writeCapable ){
      coral::ConnectionService connServ;
      std::pair<std::string,std::string> fullConnectionPars = getRealConnectionString( connectionString, transactionId );
      if( !fullConnectionPars.second.empty() ) 
	for( auto tableName : m_refreshtablelist ) connServ.webCacheControl().refreshTable( fullConnectionPars.second, tableName );

      boost::shared_ptr<coral::ISessionProxy> coralSession( connServ.connect( fullConnectionPars.first, 
									      writeCapable?coral::Update:coral::ReadOnly ) );
      return Session( coralSession, connectionString );
    }

    Session ConnectionPool::createSession( const std::string& connectionString, bool writeCapable ){
      return createSession( connectionString, "", writeCapable );
    }
      
    Session ConnectionPool::createReadOnlySession( const std::string& connectionString, const std::string& transactionId ){
      return createSession( connectionString, transactionId );
    }
    void ConnectionPool::setMessageVerbosity( coral::MsgLevel level ){
      m_messageLevel = level;
    }
    


  }
}
