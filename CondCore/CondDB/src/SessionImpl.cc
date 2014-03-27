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
	cond::DbConnection oraConnection;
	cond::DbSession oraSession =  oraConnection.createSession();
	oraSession.open( coralSession, connectionString ); 
	transaction.reset( new OraTransaction( oraSession ) );
	oraSession.transaction().start( readOnly );
	iovSchemaHandle.reset( new OraIOVSchema( oraSession ) );
	gtSchemaHandle.reset( new OraGTSchema( oraSession ) );  		       
	if( !iovSchemaHandle->exists() && !gtSchemaHandle->exists() ){
	  std::unique_ptr<IIOVSchema> iovSchema( new IOVSchema( coralSession->nominalSchema() ) );
	  std::unique_ptr<IGTSchema> gtSchema( new GTSchema( coralSession->nominalSchema() ) );
	  if( iovSchema->exists() ){
	    iovSchemaHandle = std::move(iovSchema);
	    gtSchemaHandle = std::move(gtSchema);
	    transaction.reset( new CondDBTransaction( coralSession ) );
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
