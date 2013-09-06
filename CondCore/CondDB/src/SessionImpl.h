#ifndef CondCore_CondDB_SessionImpl_h
#define CondCore_CondDB_SessionImpl_h

#include "CondCore/CondDB/interface/Configuration.h"
//#include "CondCore/CondDB/interface/Session.h"
//
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
//
#include <memory>
#include <boost/shared_ptr.hpp>

namespace coral {
  class ISessionProxy;
  class ISchema;
}

namespace conddb {

  struct TransactionCache {
    bool iovDbExists = false;
    bool iovDbOpen = false;
    bool gtDbExists = false;
    bool gtDbOpen = false;
  };

  class SessionImpl {
  public:
    SessionImpl();

    // session operation
    void connect( const std::string& connectionString, bool readOnly=true );
    void connect( const std::string& connectionString, const std::string& transactionId, bool readOnly=true );
    // TO BE REMOVED AFTER THE TRANSITION
    void connect( boost::shared_ptr<coral::ISessionProxy>& coralSession );

    void disconnect();
    bool isActive() const;
    void startTransaction( bool readOnly=true );
    void commitTransaction();
    void rollbackTransaction();
    bool isTransactionActive() const;

    coral::ISchema& coralSchema();
    
  public:
    conddb::Configuration configuration;
    coral::ConnectionService connectionService;   
    boost::shared_ptr<coral::ISessionProxy> coralSession;
    std::unique_ptr<TransactionCache> transactionCache;
  };
}

#endif

