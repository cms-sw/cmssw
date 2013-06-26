#ifndef INCLUDE_ORA_CONNECTIONPOOL_H
#define INCLUDE_ORA_CONNECTIONPOOL_H

//
#include <map>
// externals
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include "RelationalAccess/ConnectionService.h"

namespace coral {
  class ISessionProxy;
}

namespace ora {

  class SharedSession {
    public:
    SharedSession();
    SharedSession(boost::shared_ptr<coral::ISessionProxy>& coralSession);
    SharedSession( const SharedSession& rhs );
    ~SharedSession();
    SharedSession& operator=( const SharedSession& rhs );
    bool isValid();
    coral::ISessionProxy& get();
    void close();
    private:
    boost::shared_ptr<coral::ISessionProxy> m_proxy;
  };  

  /// To be moved in DBCommon, has to serve also the pure coral use case. 
  class ConnectionPool {

    public:

    ConnectionPool();

    virtual ~ConnectionPool();

    coral::IConnectionService& connectionService();
    coral::IConnectionServiceConfiguration& configuration();

    SharedSession connect( const std::string& connectionString, coral::AccessMode accessMode );
    SharedSession connect( const std::string& connectionString, const std::string& asRole, coral::AccessMode accessMode );

    private:

    static std::string lookupString( const std::string& connectionString, coral::AccessMode accessMode );
    static std::string lookupString( const std::string& connectionString, const std::string& role, coral::AccessMode accessMode );

    private:
    
    coral::ConnectionService m_connectionService;

    std::map<std::string,boost::weak_ptr<coral::ISessionProxy> > m_sessions;
  };
  

}

#endif
