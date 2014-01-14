#include "CondCore/CondDB/interface/Exception.h"
#include "SessionImpl.h"
#include "DbConnectionString.h"
// for the ORA bridge
#include "OraDbSchema.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
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

    SessionImpl::SessionImpl():
      coralSession(){
    }

    SessionImpl::SessionImpl( boost::shared_ptr<coral::ISessionProxy>& session, const std::string& connectionStr ):
      coralSession( session ),
      connectionString( connectionStr ){
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
	transaction.reset( new CondDBTransaction( coralSession ) );
	coralSession->transaction().start( readOnly );
	iovSchemaHandle.reset( new IOVSchema( coralSession->nominalSchema() ) );
	gtSchemaHandle.reset( new GTSchema( coralSession->nominalSchema() ) );  		       
	std::unique_ptr<IIOVSchema> iovSchema( new IOVSchema( coralSession->nominalSchema() ) );
	std::unique_ptr<IGTSchema> gtSchema( new GTSchema( coralSession->nominalSchema() ) );
	if( !iovSchemaHandle->exists() ){
	  cond::DbConnection oraConnection;
	  cond::DbSession oraSession =  oraConnection.createSession();
	  oraSession.open( coralSession, connectionString ); 
	  std::unique_ptr<IIOVSchema> oraIovSchema( new OraIOVSchema( oraSession ) );
	  std::unique_ptr<IGTSchema> oraGtSchema( new OraGTSchema( oraSession ) );
	  oraSession.transaction().start( readOnly );
	  // try if it is an old iov or GT schema
	  if( oraIovSchema->exists() ||  oraGtSchema->exists() ){
	    iovSchemaHandle = std::move(oraIovSchema);
	    gtSchemaHandle = std::move(oraGtSchema);
	    transaction.reset( new OraTransaction( oraSession ) );
	  } 
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
    
    bool SessionImpl::isTransactionActive() const {
      if( !transaction ) return false;
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

    void SessionImpl::openGTDb(){
      if(!transaction.get()) throwException( "The transaction is not active.","SessionImpl::open" );
      if( !transaction->gtDbOpen ){
	transaction->gtDbExists = gtSchemaHandle->exists();
	transaction->gtDbOpen = true;
      }
      if( !transaction->gtDbExists ){
	throwException( "GT Database does not exist.","SessionImpl::openGTDb");
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
