#ifndef ConditionDatabase_Configuration_h
#define ConditionDatabase_Configuration_h

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
  class DbConnectionConfiguration;
}

namespace cond {

  namespace persistency {
    // 
    enum DbAuthenticationSystem { UndefinedAuthentication=0,CondDbKey, CoralXMLFile };

    class SessionConfiguration {
    public:
      SessionConfiguration();
      ~SessionConfiguration();
      
      void setMessageVerbosity( coral::MsgLevel level );
      void setAuthenticationPath( const std::string& p );
      void setAuthenticationSystem( int authSysCode );
      void setLogging( bool flag );   
      bool isLoggingEnabled() const;
      void setParameters( const edm::ParameterSet& connectionPset );
      void configure( coral::IConnectionServiceConfiguration& coralConfig);
      // to be removed after the transition
      void configure( cond::DbConnectionConfiguration& oraConfiguration );
      bool isConfigured() const;
    private:
      std::string m_authPath;
      int m_authSys;
      coral::MsgLevel m_messageLevel;
      bool m_logging;
      // this one has to be moved!
      cond::CoralServiceManager* m_pluginManager; 
      bool m_configured = false;   
    };
  }
}

#endif

