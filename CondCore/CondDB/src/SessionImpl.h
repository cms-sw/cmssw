#ifndef CondCore_CondDB_SessionImpl_h
#define CondCore_CondDB_SessionImpl_h

#include "CondCore/CondDB/interface/Configuration.h"
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
      explicit SessionImpl( boost::shared_ptr<coral::ISessionProxy>& session );
      ~SessionImpl();
      
      void close();
      bool isActive() const;
      void startTransaction( bool readOnly=true );
      void commitTransaction();
      void rollbackTransaction();
      bool isTransactionActive() const;
      
      coral::ISchema& coralSchema();
      
    public:
      // allows for session shared among more services. To be changed to unique_ptr when we stop needing this feature.
      boost::shared_ptr<coral::ISessionProxy> coralSession;
      std::unique_ptr<TransactionCache> transactionCache;
      size_t transactionClients = 0;
    };

  }

}

#endif

