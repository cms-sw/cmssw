#ifndef COND_DBCommon_DbConnectionConfiguration_h
#define COND_DBCommon_DbConnectionConfiguration_h
//
// Package:    CondCore/DBCommon
// Class:      DbConnectionConfiguration
//
/**\class ConnectionConfiguration ConnectionConfiguration.h CondCore/DBCommon/interface/ConnectionConfiguration.h
 Description: set cofiguration parameters of the session
*/
//
//
#include <string>
// coral includes
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/IMonitoring.h"

namespace coral {
  class IConnectionServiceConfiguration;
}

namespace edm{
  class ParameterSet;
}

namespace cond{
  class CoralServiceManager;

  enum DbConfigurationDefaults { CoralDefaults, CmsDefaults, ProdDefaults, ToolDefaults, WebDefaults};

  enum DbAuthenticationSystem { UndefinedAuthentication=0,CondDbKey, CoralXMLFile };
  
  class DbConnectionConfiguration{
  public:
    static std::vector<DbConnectionConfiguration>& defaultConfigurations();
  public:
    DbConnectionConfiguration();
    DbConnectionConfiguration( bool connectionSharing,
                               int connectionTimeOut,
                               bool readOnlySessionOnUpdateConnections,
                               int connectionRetrialPeriod,
                               int connectionRetrialTimeOut,
                               bool poolAutomaticCleanUp,
                               const std::string& authenticationPath,
                               const std::string& transactionId,	
                               coral::MsgLevel msgLev,
                               coral::monitor::Level monitorLev,
                               bool SQLMonitoring );

    DbConnectionConfiguration( const DbConnectionConfiguration& rhs);
    ~DbConnectionConfiguration();
    DbConnectionConfiguration& operator=( const DbConnectionConfiguration& rhs);
    // configuration from edm parameter set
    void setParameters( const edm::ParameterSet& connectionPset );
    // configuration for individual connection parameters
    void setConnectionSharing( bool flag );
    void setConnectionTimeOut( int timeOut );
    void setReadOnlySessionOnUpdateConnections( bool flag );
    void setConnectionRetrialPeriod( int period );
    void setConnectionRetrialTimeOut( int timeout );
    void setPoolAutomaticCleanUp( bool flag );
    // authentication 
    void setAuthenticationPath( const std::string& p );
    void setAuthenticationSystem( int authSysCode );
    // transaction Id for multijob (used by frontier)
    void setTransactionId( std::string const & tid);
    // message level
    void setMessageLevel( coral::MsgLevel l );
    // monitoring level
    void setMonitoringLevel( coral::monitor::Level l );    
    // SQL monitoring
    void setSQLMonitoring( bool flag );
    // force the coral configuration
    void configure( coral::IConnectionServiceConfiguration& coralConfig) const;
    // getters
    bool isConnectionSharingEnabled() const;
    int connectionTimeOut() const;
    bool isReadOnlySessionOnUpdateConnectionEnabled() const;
    int connectionRetrialPeriod() const;
    int connectionRetrialTimeOut() const;
    bool isPoolAutomaticCleanUpEnabled() const;
    const std::string& authenticationPath() const;
    const std::string& transactionId() const;
    coral::MsgLevel messageLevel() const;
    bool isSQLMonitoringEnabled() const;
    private:
    std::pair<bool,bool> m_connectionSharing;
    std::pair<bool,int> m_connectionTimeOut;
    std::pair<bool,bool> m_readOnlySessionOnUpdateConnections;
    std::pair<bool,int> m_connectionRetrialPeriod;
    std::pair<bool,int> m_connectionRetrialTimeOut;
    std::pair<bool,bool> m_poolAutomaticCleanUp;
    std::string m_authPath;
    int m_authSys;
    std::string m_transactionId;
    coral::MsgLevel m_messageLevel;
    coral::monitor::Level m_monitoringLevel;  
    //int m_idleConnectionCleanupPeriod;

    bool m_SQLMonitoring;
    CoralServiceManager* m_pluginManager;    
  };
}
#endif
