#ifndef CondCore_CondDB_SessionImpl_h
#define CondCore_CondDB_SessionImpl_h

#include "CondCore/CondDB/interface/Configuration.h"
//
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
//
#include <memory>
// temporarely
#include <boost/shared_ptr.hpp>

namespace coral {
  class ISessionProxy;
  class ISchema;
}

namespace cond {

  namespace persistency {

    struct TransactionCache {
      bool iovDbExists = false;
      bool iovDbOpen = false;
      bool gtDbExists = false;
      bool gtDbOpen = false;
    };
    
    class SessionImpl {
    public:
      SessionImpl();
      ~SessionImpl();
      
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
      SessionConfiguration configuration;
      coral::ConnectionService connectionService;   
      // allows for session shared among more services. To be changed to unique_ptr when we stop needing this feature.
      boost::shared_ptr<coral::ISessionProxy> coralSession;
      std::unique_ptr<TransactionCache> transactionCache;
      size_t transactionClients = 0;
    };

  }

}

#endif

