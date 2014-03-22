#ifndef CondCore_CondDB_SessionImpl_h
#define CondCore_CondDB_SessionImpl_h

#include "CondCore/CondDB/interface/Types.h"
#include "IOVSchema.h"
#include "GTSchema.h"
//
#include "CondCore/DBCommon/interface/DbSession.h"
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

    class ITransaction {
    public:
      virtual ~ITransaction(){}
      virtual void commit() = 0;
      virtual void rollback() = 0;
      virtual bool isActive() = 0;
      bool iovDbExists = false;
      bool iovDbOpen = false;
      bool gtDbExists = false;
      bool gtDbOpen = false;
      bool isOra = false;
      size_t clients = 0;
    };

    BackendType checkBackendType( boost::shared_ptr<coral::ISessionProxy>& session, 
				  const std::string& connectionString );
    
    class SessionImpl {
    public:
      typedef enum { THROW, DO_NOT_THROW, CREATE } FailureOnOpeningPolicy;
    public:
      SessionImpl();
      SessionImpl( boost::shared_ptr<coral::ISessionProxy>& session, 
		   const std::string& connectionString, 
		   BackendType backType );

      ~SessionImpl();
      
      void close();
      bool isActive() const;
      void startTransaction( bool readOnly=true );
      void commitTransaction();
      void rollbackTransaction();
      bool isTransactionActive( bool deep=true ) const;

      void openIovDb( FailureOnOpeningPolicy policy = THROW );
      void openGTDb();
      IIOVSchema& iovSchema();
      IGTSchema& gtSchema();
      // only for the bridging...
      bool isOra();
      
    public:
      // allows for session shared among more services. To be changed to unique_ptr when we stop needing this feature.
      boost::shared_ptr<coral::ISessionProxy> coralSession;
      // not really useful outside the ORA bridging...
      std::string connectionString;
      BackendType theBackendType;
      std::unique_ptr<ITransaction> transaction;
      std::unique_ptr<IIOVSchema> iovSchemaHandle; 
      std::unique_ptr<IGTSchema> gtSchemaHandle; 
    };

  }

}

#endif

