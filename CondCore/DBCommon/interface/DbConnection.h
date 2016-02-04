#ifndef COND_DBCommon_DbConnection_h
#define COND_DBCommon_DbConnection_h

#include <string>
#include <boost/shared_ptr.hpp>
#include "DbConnectionConfiguration.h"
#include "DbSession.h"
//
// Package:    CondCore/DBCommon
// Class:      DbConnection
//
/**\class DbConnection DbConnection.h CondCore/DBCommon/interface/DbConnection.h
 Description: Class to prepare database connection setup
*/

namespace coral {
  class ConnectionService;
  class ISessionProxy;
  class IMonitoringReporter;
  class IWebCacheControl;
}

namespace edm{
  class ParameterSet;
}

namespace ora {
  class ConnectionPool;
}

namespace cond{

  /*
  **/
  class DbConnection{
  public:
    DbConnection();
    
    DbConnection(const DbConnection& conn);
    
    virtual ~DbConnection();

    DbConnection& operator=(const DbConnection& conn);
    
    void configure();    
    
    void configure( cond::DbConfigurationDefaults defaultItem );    
    
    void configure( const edm::ParameterSet& connectionPset );
    
    void close();

    bool isOpen() const;

    DbSession createSession() const;

    DbConnectionConfiguration & configuration();
    DbConnectionConfiguration const & configuration() const;

    boost::shared_ptr<ora::ConnectionPool> connectionPool() const;

    coral::IConnectionService& connectionService() const;

    const coral::IMonitoringReporter& monitoringReporter() const;

    coral::IWebCacheControl& webCacheControl() const;

    private:
    class ConnectionImpl;
    
    private:
    boost::shared_ptr<ConnectionImpl> m_implementation;

  };
}
#endif
// DBCONNECTION_H
