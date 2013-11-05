#include "CondCore/CondDB/interface/Session.h"
#include "SessionImpl.h"
//
#include <openssl/sha.h>

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

    void Session::open( const std::string& connectionString, bool readOnly ){
      m_session->connect( connectionString, readOnly );
    } 

    // TO BE REMOVED AFTER THE TRANSITION
    void Session::open( boost::shared_ptr<coral::ISessionProxy> coralSession ){
      m_session->connect( coralSession );
    }

    void Session::close(){
      m_session->disconnect();
    }

    SessionConfiguration& Session::configuration(){
      return m_session->configuration;
    }

    Transaction& Session::transaction(){
      return m_transaction;
    }

    //
    bool Session::existsDatabase(){
      m_session->openIovDb( SessionImpl::DO_NOT_THROW );
      return m_session->transactionCache->iovDbExists;
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

    IOVEditor Session::createIov( const std::string& payloadType, const std::string& tag, cond::TimeType timeType, cond::SynchronizationType synchronizationType ){
      m_session->openIovDb( SessionImpl::CREATE );
      if( m_session->iovSchema().tagTable().select( tag ) ) 
	throwException( "The specified tag \""+tag+"\" already exist in the database.","Session::createIov");
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
    
    cond::Hash makeHash( const std::string& objectType, const cond::Binary& data ){
      SHA_CTX ctx;
      if( !SHA1_Init( &ctx ) ){
	throwException( "SHA1 initialization error.","Session::makeHash");
      }
      if( !SHA1_Update( &ctx, objectType.c_str(), objectType.size() ) ){
	throwException( "SHA1 processing error (1).","Session::makeHash");
      }
      if( !SHA1_Update( &ctx, data.data(), data.size() ) ){
	throwException( "SHA1 processing error (2).","Session::makeHash");
      }
      unsigned char hash[SHA_DIGEST_LENGTH];
      if( !SHA1_Final(hash, &ctx) ){
	throwException( "SHA1 finalization error.","Session::makeHash");
      }
      
      char tmp[SHA_DIGEST_LENGTH*2+1];
      // re-write bytes in hex
      for (unsigned int i = 0; i < 20; i++) {                                                                                                        
	::sprintf(&tmp[i * 2], "%02x", hash[i]);                                                                                                 
      }                                                                                                                                              
      tmp[20*2] = 0;                                                                                                                                 
      return tmp;                                                                                                                                    
    }
    
    cond::Hash Session::storePayloadData( const std::string& payloadObjectType, 
					  const cond::Binary& payloadData, 
					  const boost::posix_time::ptime& creationTime ){
      m_session->openIovDb( SessionImpl::CREATE );
      cond::Hash payloadHash = makeHash( payloadObjectType, payloadData );
      // the check on the hash existance is only required to avoid the error message printing in SQLite! once this is removed, this check is useless... 
      if( !m_session->iovSchema().payloadTable().select( payloadHash ) ){
	m_session->iovSchema().payloadTable().insert( payloadHash, payloadObjectType, payloadData, creationTime );
      }
      return payloadHash;
    }
    
    bool Session::fetchPayloadData( const cond::Hash& payloadHash,
				    std::string& payloadType, 
				    cond::Binary& payloadData ){
      m_session->openIovDb();
      return m_session->iovSchema().payloadTable().select( payloadHash, payloadType, payloadData );
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
    
  }
}
