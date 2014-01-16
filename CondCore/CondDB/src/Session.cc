#include "CondCore/CondDB/interface/Session.h"
#include "SessionImpl.h"
//

namespace cond {

  namespace persistency {

    Transaction::Transaction( SessionImpl& session ):
      m_session( &session ){
    }

    Transaction::Transaction( const Transaction& rhs ):
      m_session( rhs.m_session ){
    }

    Transaction& Transaction::operator=( const Transaction& rhs ){
      m_session = rhs.m_session;
      return *this;
    }
    
    void Transaction::start( bool readOnly ){
      m_session->startTransaction( readOnly );
    }

    void Transaction::commit(){
      m_session->commitTransaction();
    }

    void Transaction::rollback(){
      m_session->rollbackTransaction();
    }

    bool Transaction::isActive(){
      return m_session->isTransactionActive();
    }

    Session::Session():
      m_session( new SessionImpl ),
      m_transaction( *m_session ){
    }
    
    Session::Session( const std::shared_ptr<SessionImpl>& sessionImpl ):
      m_session( sessionImpl ),
      m_transaction( *m_session ){      
    }

    Session::Session( boost::shared_ptr<coral::ISessionProxy>& session, const std::string& connectionString ):
      m_session( new SessionImpl( session, connectionString ) ),
      m_transaction( *m_session ){
    }

    Session::Session( const Session& rhs ):
      m_session( rhs.m_session ),
      m_transaction( rhs.m_transaction ){
    }
    
    Session::~Session(){
    }

    Session& Session::operator=( const Session& rhs ){
      m_session = rhs.m_session;
      m_transaction = rhs.m_transaction;
      return *this;
    }

    void Session::close(){
      m_session->close();
    }

    Transaction& Session::transaction(){
      return m_transaction;
    }

    //
    bool Session::existsDatabase(){
      m_session->openIovDb( SessionImpl::DO_NOT_THROW );
      return m_session->transaction->iovDbExists;
    }
    
    //
    void Session::createDatabase(){
      m_session->openIovDb( SessionImpl::CREATE );
    }

    IOVProxy Session::readIov( const std::string& tag, bool full ){
      m_session->openIovDb();
      IOVProxy proxy( m_session );
      proxy.load( tag, full );
      return proxy;
    }

    bool Session::existsIov( const std::string& tag ){
      m_session->openIovDb();
      return m_session->iovSchema().tagTable().select( tag );
    }
    
    IOVProxy Session::iovProxy(){
      m_session->openIovDb();
      IOVProxy proxy( m_session );
      return proxy;
    }

    IOVEditor Session::createIov( const std::string& payloadType, const std::string& tag, cond::TimeType timeType, 
				  cond::SynchronizationType synchronizationType ){
      m_session->openIovDb( SessionImpl::CREATE );
      if( m_session->iovSchema().tagTable().select( tag ) ) 
	throwException( "The specified tag \""+tag+"\" already exist in the database.","Session::createIov");
      IOVEditor editor( m_session, tag, timeType, payloadType, synchronizationType );
      return editor;
    }

    IOVEditor Session::createIovForPayload( const Hash& payloadHash, const std::string& tag, cond::TimeType timeType,
					    cond::SynchronizationType synchronizationType ){
      m_session->openIovDb( SessionImpl::CREATE );
      if( m_session->iovSchema().tagTable().select( tag ) ) 
	throwException( "The specified tag \""+tag+"\" already exist in the database.","Session::createIovForPayload");
      std::string payloadType("");
      if( !m_session->iovSchema().payloadTable().getType( payloadHash, payloadType ) )
	throwException( "The specified payloadId \""+payloadHash+"\" does not exist in the database.","Session::createIovForPayload");
      IOVEditor editor( m_session, tag, timeType, payloadType, synchronizationType );
      return editor;      
    }
    
    IOVEditor Session::editIov( const std::string& tag ){
      m_session->openIovDb();
      IOVEditor editor( m_session );
      editor.load( tag );
      return editor;
    }
    
    GTEditor Session::createGlobalTag( const std::string& name ){
      m_session->openGTDb();
      if( m_session->gtSchema().gtTable().select( name ) ) 
	throwException( "The specified Global Tag \""+name+"\" already exist in the database.","Session::createGlobalTag");
      GTEditor editor( m_session, name );
      return editor;
    }
    
    GTEditor Session::editGlobalTag( const std::string& name ){
      m_session->openGTDb();
      GTEditor editor( m_session );
      editor.load( name );
      return editor;
    }
    
    GTProxy Session::readGlobalTag( const std::string& name ){
      m_session->openGTDb();
      GTProxy proxy( m_session );
      proxy.load( name );
      return proxy;
    }
    
    GTProxy Session::readGlobalTag( const std::string& name, const std::string& preFix, const std::string& postFix  ){
      m_session->openGTDb();
      GTProxy proxy( m_session );
      proxy.load( name, preFix, postFix );
      return proxy;
    }

    cond::Hash Session::storePayloadData( const std::string& payloadObjectType, 
					  const cond::Binary& payloadData, 
					  const boost::posix_time::ptime& creationTime ){
      m_session->openIovDb( SessionImpl::CREATE );
      return m_session->iovSchema().payloadTable().insertIfNew( payloadObjectType, payloadData, creationTime );
    }
    
    bool Session::fetchPayloadData( const cond::Hash& payloadHash,
				    std::string& payloadType, 
				    cond::Binary& payloadData ){
      m_session->openIovDb();
      return m_session->iovSchema().payloadTable().select( payloadHash, payloadType, payloadData );
    }

    bool Session::isOraSession(){
      return m_session->isOra();
    }
    
    bool Session::checkMigrationLog( const std::string& sourceAccount, const std::string& sourceTag, std::string& destTag ){
      if(! m_session->iovSchema().tagMigrationTable().exists() ) m_session->iovSchema().tagMigrationTable().create();
      //throwException( "Migration Log Table does not exist in this schema.","Session::checkMigrationLog");
      return m_session->iovSchema().tagMigrationTable().select( sourceAccount, sourceTag, destTag );
    }
    
    void Session::addToMigrationLog( const std::string& sourceAccount, const std::string& sourceTag, const std::string& destTag ){
      if(! m_session->iovSchema().tagMigrationTable().exists() ) m_session->iovSchema().tagMigrationTable().create();
      m_session->iovSchema().tagMigrationTable().insert( sourceAccount, sourceTag, destTag, 
							 boost::posix_time::microsec_clock::universal_time() );
    }

    std::string Session::connectionString(){
      return m_session->connectionString;
    }

    coral::ISessionProxy& Session::coralSession(){
      if( !m_session->coralSession.get() ) throwException( "The session is not active.","Session::coralSession");
      return *m_session->coralSession; 
    }

    coral::ISchema& Session::nominalSchema(){
      return coralSession().nominalSchema();
    }

    TransactionScope::TransactionScope( Transaction& transaction ):
      m_transaction(transaction),m_status(true){
      m_status = !m_transaction.isActive();
    }

    TransactionScope::~TransactionScope(){
      if(!m_status && m_transaction.isActive() ) {
	m_transaction.rollback();
      }
    }

    void TransactionScope::start( bool readOnly ){
      m_transaction.start( readOnly );
      m_status = false;
    }

    void TransactionScope::commit(){
      m_transaction.commit();
      m_status = true;
    }
    
    void TransactionScope::close(){
      m_status = true;
    }
    
  }
}
