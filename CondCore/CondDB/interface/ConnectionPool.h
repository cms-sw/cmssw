#ifndef ConditionDatabase_ConnectionPool_h
#define ConditionDatabase_ConnectionPool_h

#include "CondCore/CondDB/interface/Session.h"
//
#include <string>
#include <memory>
//
#include "CoralBase/MessageStream.h"

namespace edm {
  class ParameterSet;
}

namespace coral {
  class IConnectionServiceConfiguration;
  class ISessionProxy;
}

namespace cond {
  class CoralServiceManager;
}

namespace cond {

  namespace persistency {
    // 
    enum DbAuthenticationSystem { UndefinedAuthentication=0,CondDbKey, CoralXMLFile };

    // a wrapper for the coral connection service.  
    class ConnectionPool {
    public:
      ConnectionPool();
      ~ConnectionPool();

      void setMessageVerbosity( coral::MsgLevel level );
      void setAuthenticationPath( const std::string& p );
      void setAuthenticationSystem( int authSysCode );
      void setFrontierSecurity( const std::string& signature );
      void setLogging( bool flag );   
      bool isLoggingEnabled() const;
      void setParameters( const edm::ParameterSet& connectionPset );
      void configure();
      Session createSession( const std::string& connectionString, bool writeCapable = false );
      Session createReadOnlySession( const std::string& connectionString, const std::string& transactionId );
      boost::shared_ptr<coral::ISessionProxy> createCoralSession( const std::string& connectionString, bool writeCapable = false );
      
    private:
      boost::shared_ptr<coral::ISessionProxy> createCoralSession( const std::string& connectionString, 
                                                                  const std::string& transactionId, 
                                                                  bool writeCapable = false );
      Session createSession( const std::string& connectionString, 
                             const std::string& transactionId, 
                             bool writeCapable = false );
      void configure( coral::IConnectionServiceConfiguration& coralConfig );
    private:
      std::string m_authPath = std::string( "" );
      int m_authSys = 0;
      coral::MsgLevel m_messageLevel = coral::Error;
      bool m_loggingEnabled = false;
      //The frontier security option is turned on for all sessions
      //usig this wrapper of the CORAL connection setup for configuring the server access
      std::string m_frontierSecurity = std::string( "" );
      // this one has to be moved!
      cond::CoralServiceManager* m_pluginManager = nullptr; 
      std::map<std::string,int> m_dbTypes;
    };
  }
}

#endif

