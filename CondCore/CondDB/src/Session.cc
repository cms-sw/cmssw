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
      m_session->openDb();
    }

    IOVProxy Session::readIov( const std::string& tag, bool full ){
      m_session->openIovDb();
      IOVProxy proxy( m_session );
      proxy.load( tag, full );
      return proxy;
    }

    IOVProxy  Session::readIov( const std::string& tag,
                                const boost::posix_time::ptime& snapshottime,
                                bool full ){
      m_session->openIovDb();
      IOVProxy proxy( m_session );
      proxy.load( tag, snapshottime, full );
      return proxy;
    }

    bool Session::existsIov( const std::string& tag ){
      m_session->openIovDb();
      return m_session->iovSchema().tagTable().select( tag );
    }

    bool Session::getTagInfo( const std::string& tag, 
			      cond::Tag_t& info ){
      m_session->openIovDb();
      std::string description;
      return m_session->iovSchema().tagTable().select( tag, info.timeType, info.payloadType, info.synchronizationType,
							    info.endOfValidity, description, info.lastValidatedTime );      
    }
    
    IOVProxy Session::iovProxy(){
      m_session->openIovDb();
      IOVProxy proxy( m_session );
      return proxy;
    }

    bool Session::getIovRange( const std::string& tag, cond::Time_t begin, cond::Time_t end, 
			    std::vector<std::tuple<cond::Time_t,cond::Hash> >& range ){
      m_session->openIovDb();
      boost::posix_time::ptime snapshotTime;
      return m_session->iovSchema().iovTable().getRange( tag, begin, end, snapshotTime, range );
    }

    IOVEditor Session::createIov( const std::string& payloadType, const std::string& tag, cond::TimeType timeType, 
				  cond::SynchronizationType synchronizationType ){
      m_session->openDb();
      if( m_session->iovSchema().tagTable().select( tag ) ) 
	throwException( "The specified tag \""+tag+"\" already exist in the database.","Session::createIov");
      IOVEditor editor( m_session, tag, timeType, payloadType, synchronizationType );
      return editor;
    }

    IOVEditor Session::createIov( const std::string& payloadType, 
				  const std::string& tag, 
				  cond::TimeType timeType,
				  cond::SynchronizationType synchronizationType,
				  const boost::posix_time::ptime& creationTime ){
      m_session->openDb();
      if( m_session->iovSchema().tagTable().select( tag ) ) 
	throwException( "The specified tag \""+tag+"\" already exist in the database.","Session::createIov");
      IOVEditor editor( m_session, tag, timeType, payloadType, synchronizationType, creationTime );
      return editor;
    }

    IOVEditor Session::createIovForPayload( const Hash& payloadHash, const std::string& tag, cond::TimeType timeType,
					    cond::SynchronizationType synchronizationType ){
      m_session->openDb();
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

    void Session::clearIov( const std::string& tag ){
      m_session->openIovDb();
      m_session->iovSchema().iovTable().erase( tag );      
    }

    bool Session::existsGlobalTag( const std::string& name ){
      m_session->openGTDb();
      return m_session->gtSchema().gtTable().select( name );    
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
					  const std::pair<Binary,Binary>& payloadAndStreamerInfoData,
					  const boost::posix_time::ptime& creationTime ){
      m_session->openDb();
      return m_session->iovSchema().payloadTable().insertIfNew( payloadObjectType, payloadAndStreamerInfoData.first, 
								payloadAndStreamerInfoData.second, creationTime );
    }
    
    bool Session::fetchPayloadData( const cond::Hash& payloadHash,
				    std::string& payloadType, 
				    cond::Binary& payloadData,
				    cond::Binary& streamerInfoData ){
      m_session->openIovDb();
      return m_session->iovSchema().payloadTable().select( payloadHash, payloadType, payloadData, streamerInfoData );
    }

    RunInfoProxy Session::getRunInfo( cond::Time_t start, cond::Time_t end ){
      if(!m_session->transaction.get()) 
	throwException( "The transaction is not active.","Session::getRunInfo" );
      RunInfoProxy proxy( m_session );
      proxy.load( start, end );
      return proxy;
    }

    RunInfoEditor Session::editRunInfo(){
      RunInfoEditor editor(m_session);
      editor.init();
      return editor;
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
