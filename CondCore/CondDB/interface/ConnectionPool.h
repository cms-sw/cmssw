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
      void setLogging( bool flag );   
      bool isLoggingEnabled() const;
      void setParameters( const edm::ParameterSet& connectionPset );
      void configure();
      Session createSession( const std::string& connectionString, bool writeCapable=false, BackendType backType=DEFAULT_DB );
      Session createReadOnlySession( const std::string& connectionString, const std::string& transactionId );
      
    private:
      Session createSession( const std::string& connectionString, 
			     const std::string& transactionId, 
			     bool writeCapable=false, 
			     BackendType backType=DEFAULT_DB );
      void configure( coral::IConnectionServiceConfiguration& coralConfig);
    private:
      std::string m_authPath;
      int m_authSys = 0;
      coral::MsgLevel m_messageLevel = coral::Error;
      bool m_loggingEnabled = false;
      // this one has to be moved!
      cond::CoralServiceManager* m_pluginManager = 0; 
      std::map<std::string,int> m_dbTypes;
    };
  }
}

#endif

