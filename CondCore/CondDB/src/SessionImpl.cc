#include "CondCore/CondDB/interface/Exception.h"
#include "SessionImpl.h"
#include "DbConnectionString.h"
// for the ORA bridge
#include "OraDbSchema.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
//-ap: also to be removed when ORA goes:
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
//
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"

namespace cond {

  namespace persistency {

    class CondDBTransaction : public ITransaction {
    public:
      CondDBTransaction( const boost::shared_ptr<coral::ISessionProxy>& coralSession ):
	m_session( coralSession ){
      }
      virtual ~CondDBTransaction(){}
     
      void commit(){
	m_session->transaction().commit();
      }
      
      void rollback(){
	m_session->transaction().rollback();
      }

      bool isActive(){
	return m_session->transaction().isActive();
      }
    private: 
      boost::shared_ptr<coral::ISessionProxy> m_session;
    };

    class OraTransaction : public ITransaction {
    public:
      OraTransaction( const cond::DbSession& session ):
	m_session( session ){
	isOra = true;
      }
      virtual ~OraTransaction(){}

      void commit(){
	m_session.transaction().commit();
      }

      void rollback(){
	m_session.transaction().rollback();
      }
      bool isActive() {
	return m_session.transaction().isActive();
      }
    private:
      cond::DbSession m_session;
    };

    BackendType checkBackendType( boost::shared_ptr<coral::ISessionProxy>& coralSession, 
				  const std::string& connectionString ){
      BackendType ret = UNKNOWN_DB;
      cond::DbSession oraSession;
      oraSession.open( coralSession, connectionString ); 
      oraSession.transaction().start( true );
      std::unique_ptr<IIOVSchema> iovSchemaHandle( new OraIOVSchema( oraSession ) );
      std::unique_ptr<IGTSchema> gtSchemaHandle( new OraGTSchema( oraSession ) );  		       
      if( !iovSchemaHandle->exists() && !gtSchemaHandle->exists() ){
	iovSchemaHandle.reset( new IOVSchema( coralSession->nominalSchema() ) );
        ret = COND_DB;
      } else {
	ret = ORA_DB;
      }
      oraSession.transaction().commit();
      return ret;      
    }

    SessionImpl::SessionImpl():
      coralSession(),
      theBackendType( UNKNOWN_DB ){
    }

    SessionImpl::SessionImpl( boost::shared_ptr<coral::ISessionProxy>& session, 
			      const std::string& connectionStr,
			      BackendType backType ):
      coralSession( session ),
      connectionString( connectionStr ),
      theBackendType( backType ){
    }

    SessionImpl::~SessionImpl(){
      close();
    }

    void SessionImpl::close(){
      if( coralSession.get() ){
	if( coralSession->transaction().isActive() ){
	  coralSession->transaction().rollback();
	}
	coralSession.reset();
      }
      transaction.reset();
    }
    
    bool SessionImpl::isActive() const {
      return coralSession.get();
    }

    void SessionImpl::startTransaction( bool readOnly ){
      if( !transaction.get() ){ 
	if ( theBackendType == ORA_DB ) {
	  cond::DbSession oraSession;
	  oraSession.open( coralSession, connectionString ); 
	  oraSession.transaction().start( readOnly );
	  iovSchemaHandle.reset( new OraIOVSchema( oraSession ) );
	  gtSchemaHandle.reset( new OraGTSchema( oraSession ) );  		       
	  transaction.reset( new OraTransaction( oraSession ) );
	} else if ( theBackendType == COND_DB ){
	  coralSession->transaction().start( readOnly );
	  iovSchemaHandle.reset( new IOVSchema( coralSession->nominalSchema() ) );
	  gtSchemaHandle.reset( new GTSchema( coralSession->nominalSchema() ) );
	  transaction.reset( new CondDBTransaction( coralSession ) );
	} else {
	  throwException( "No valid database found.", "SessionImpl::startTransaction" );
	}
      } else {
	if(!readOnly ) throwException( "An update transaction is already active.",
				       "SessionImpl::startTransaction" );
      }
      transaction->clients++;
    }
    
    void SessionImpl::commitTransaction(){
      if( transaction ) {
	transaction->clients--;
	if( !transaction->clients ){
	  transaction->commit();
	  transaction.reset();
	  iovSchemaHandle.reset();
	  gtSchemaHandle.reset();
	}
      }
    }
    
    void SessionImpl::rollbackTransaction(){
      if( transaction ) {   
	transaction->rollback();
	transaction.reset();
	iovSchemaHandle.reset();
	gtSchemaHandle.reset();
      }
    }
    
    bool SessionImpl::isTransactionActive( bool deep ) const {
      if( !transaction ) return false;
      if( !deep ) return true;
      return transaction->isActive();
    }

    void SessionImpl::openIovDb( SessionImpl::FailureOnOpeningPolicy policy ){
      if(!transaction.get()) throwException( "The transaction is not active.","SessionImpl::openIovDb" );
      if( !transaction->iovDbOpen ){
	transaction->iovDbExists = iovSchemaHandle->exists();
	transaction->iovDbOpen = true;
      }      
      if( !transaction->iovDbExists ){
	if( policy==CREATE ){
	  iovSchemaHandle->create();
	  transaction->iovDbExists = true;
	} else {
	  if( policy==THROW) throwException( "IOV Database does not exist.","SessionImpl::openIovDb");
	}
      }
    }

    void SessionImpl::openGTDb( SessionImpl::FailureOnOpeningPolicy policy ){
      if(!transaction.get()) throwException( "The transaction is not active.","SessionImpl::open" );
      if( !transaction->gtDbOpen ){
	transaction->gtDbExists = gtSchemaHandle->exists();
	transaction->gtDbOpen = true;
      }
      if( !transaction->gtDbExists ){
        if( policy==CREATE ){
          gtSchemaHandle->create();
          transaction->gtDbExists = true;
        } else {
          if( policy==THROW) throwException( "GT Database does not exist.","SessionImpl::openGTDb");
	}
      }
    }

    void SessionImpl::openDb(){
      if(!transaction.get()) throwException( "The transaction is not active.","SessionImpl::openIovDb" );
      if( !transaction->iovDbOpen ){
        transaction->iovDbExists = iovSchemaHandle->exists();
        transaction->iovDbOpen = true;
      }
      if( !transaction->gtDbOpen ){
        transaction->gtDbExists = gtSchemaHandle->exists();
        transaction->gtDbOpen = true;
      }
      if( !transaction->iovDbExists ){
	iovSchemaHandle->create();
	transaction->iovDbExists = true;
	if( !transaction->gtDbExists ){
	  gtSchemaHandle->create();
	  transaction->gtDbExists = true;
	}
      }
    }
    
    IIOVSchema& SessionImpl::iovSchema(){
      return *iovSchemaHandle;
    }

    IGTSchema& SessionImpl::gtSchema(){
      return *gtSchemaHandle;
    }

    bool SessionImpl::isOra(){
      if(!transaction.get()) throwException( "The transaction is not active.","SessionImpl::open" );
      return transaction->isOra;
    }

  }
}
