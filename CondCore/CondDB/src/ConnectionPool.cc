#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "DbConnectionString.h"
#include "SessionImpl.h"
#include "IOVSchema.h"
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

    static const std::string POOL_IOV_TABLE_DATA("IOV_DATA");
    static const std::string ORA_IOV_TABLE_1("ORA_C_COND_IOVSEQUENCE");
    static const std::string ORA_IOV_TABLE_2("ORA_C_COND_IOVSEQU_A0");
    static const std::string ORA_IOV_TABLE_3("ORA_C_COND_IOVSEQU_A1");
   
    ConnectionPool::ConnectionPool(){
      m_pluginManager = new cond::CoralServiceManager;
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

    Session ConnectionPool::createSession( const std::string& connectionString, 
					   const std::string& transactionId, 
					   bool writeCapable,
					   BackendType backType){
      coral::ConnectionService connServ;
      std::pair<std::string,std::string> fullConnectionPars = getConnectionParams( connectionString, transactionId );
      if( !fullConnectionPars.second.empty() ) {
	// the olds formats
	connServ.webCacheControl().refreshTable( fullConnectionPars.second, POOL_IOV_TABLE_DATA );
	connServ.webCacheControl().refreshTable( fullConnectionPars.second, ORA_IOV_TABLE_1 );
	connServ.webCacheControl().refreshTable( fullConnectionPars.second, ORA_IOV_TABLE_2 );
	connServ.webCacheControl().refreshTable( fullConnectionPars.second, ORA_IOV_TABLE_3 );
	// the new schema...
	connServ.webCacheControl().setTableTimeToLive( fullConnectionPars.second, TAG::tname, 1 );
	connServ.webCacheControl().setTableTimeToLive( fullConnectionPars.second, IOV::tname, 1 );
	connServ.webCacheControl().setTableTimeToLive( fullConnectionPars.second, PAYLOAD::tname, 3 );
      }

      boost::shared_ptr<coral::ISessionProxy> coralSession( connServ.connect( fullConnectionPars.first, 
									      writeCapable?Auth::COND_WRITER_ROLE:Auth::COND_READER_ROLE,
									      writeCapable?coral::Update:coral::ReadOnly ) );
      BackendType bt;
      auto it = m_dbTypes.find( connectionString);
      if( it == m_dbTypes.end() ){
	bt = checkBackendType( coralSession, connectionString );
	if( bt == UNKNOWN_DB && writeCapable) bt = backType;
	m_dbTypes.insert( std::make_pair( connectionString, bt ) ).first;
      } else {
	bt = (BackendType) it->second;
      }
   
      std::shared_ptr<SessionImpl> impl( new SessionImpl( coralSession, connectionString, bt ) );  
      return Session( impl );
    }

    Session ConnectionPool::createSession( const std::string& connectionString, bool writeCapable, BackendType backType ){
      return createSession( connectionString, "", writeCapable, backType );
    }
      
    Session ConnectionPool::createReadOnlySession( const std::string& connectionString, const std::string& transactionId ){
      return createSession( connectionString, transactionId );
    }
    void ConnectionPool::setMessageVerbosity( coral::MsgLevel level ){
      m_messageLevel = level;
    }
    


  }
}
