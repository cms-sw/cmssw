#include "CondCore/CondDB/interface/Session.h"
#include "IOVSchema.h"
#include "GTSchema.h"
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
      openIovDb( DO_NOT_THROW );
      return m_session->transactionCache->iovDbExists;
    }
    
    //
    void Session::createDatabase(){
      openIovDb( CREATE );
    }

    void Session::openIovDb( Session::OpenFailurePolicy policy ){
      if(!m_session->transactionCache.get()) throwException( "The transaction is not active.","Session::open" );
      if( !m_session->transactionCache->iovDbOpen ){
	m_session->transactionCache->iovDbExists = cond::persistency::iovDb::exists( *m_session );
	m_session->transactionCache->iovDbOpen = true;
      }      if( !m_session->transactionCache->iovDbExists ){
	if( policy==CREATE ){
	  cond::persistency::iovDb::create( *m_session );
	  m_session->transactionCache->iovDbExists = true;
	} else {
	  if( policy==THROW) throwException( "IOV Database does not exist.","Session::openIovDb");
	}
      }
    }
    
    void Session::openGTDb(){
      if(!m_session->transactionCache.get()) throwException( "The transaction is not active.","Session::open" );
      if( !m_session->transactionCache->gtDbOpen ){
	m_session->transactionCache->gtDbExists = cond::persistency::gtDb::exists( *m_session );
	m_session->transactionCache->gtDbOpen = true;
      }
      if( !m_session->transactionCache->gtDbExists ){
	throwException( "GT Database does not exist.","Session::openIovDb");
      }
    }
    
    IOVProxy Session::readIov( const std::string& tag, bool full ){
      openIovDb();
      IOVProxy proxy( m_session );
      proxy.load( tag, full );
      return proxy;
    }

    bool Session::existsIov( const std::string& tag ){
      openIovDb();
      return TAG::select( tag, *m_session );
    }
    
    IOVProxy Session::iovProxy(){
      openIovDb();
      IOVProxy proxy( m_session );
      return proxy;
    }

    IOVEditor Session::createIov( const std::string& tag, cond::TimeType timeType, const std::string& payloadType, cond::SynchronizationType synchronizationType ){
      openIovDb( CREATE );
      if( TAG::select( tag, *m_session ) ) throwException( "The specified tag \""+tag+"\" already exist in the database.","Session::createIov");
      IOVEditor editor( m_session, tag, timeType, payloadType, synchronizationType );
      return editor;
    }
    
    IOVEditor Session::editIov( const std::string& tag ){
      openIovDb();
      IOVEditor editor( m_session );
      editor.load( tag );
      return editor;
    }
    
    GTEditor Session::createGlobalTag( const std::string& name ){
      openGTDb();
      if( GLOBAL_TAG::select( name, *m_session ) ) throwException( "The specified Global Tag \""+name+"\" already exist in the database.","Session::createGlobalTag");
      GTEditor editor( m_session, name );
      return editor;
    }
    
    GTEditor Session::editGlobalTag( const std::string& name ){
      openGTDb();
      GTEditor editor( m_session );
      editor.load( name );
      return editor;
    }
    
    GTProxy Session::readGlobalTag( const std::string& name ){
      openGTDb();
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
      openIovDb( CREATE );
      cond::Hash payloadHash = makeHash( payloadObjectType, payloadData );
      // the check on the hash existance is only required to avoid the error message printing in SQLite! once this is removed, this check is useless... 
      if( !PAYLOAD::select( payloadHash, *m_session ) ){
	PAYLOAD::insert( payloadHash, payloadObjectType, payloadData, creationTime, *m_session );
      }
      return payloadHash;
    }
    
    bool Session::fetchPayloadData( const cond::Hash& payloadHash,
				    std::string& payloadType, 
				    cond::Binary& payloadData ){
      openIovDb();
      return PAYLOAD::select( payloadHash, payloadType, payloadData, *m_session );
    }
    
    bool Session::checkMigrationLog( const std::string& sourceAccount, const std::string& sourceTag, std::string& destTag ){
      if(! TAG_MIGRATION::exists(  *m_session ) ) TAG_MIGRATION::create( *m_session );
      //throwException( "Migration Log Table does not exist in this schema.","Session::checkMigrationLog");
      return TAG_MIGRATION::select( sourceAccount, sourceTag, destTag, *m_session );
    }
    
    void Session::addToMigrationLog( const std::string& sourceAccount, const std::string& sourceTag, const std::string& destTag ){
      if(! TAG_MIGRATION::exists(  *m_session ) ) TAG_MIGRATION::create( *m_session );
      TAG_MIGRATION::insert( sourceAccount, sourceTag, destTag, boost::posix_time::microsec_clock::universal_time(), *m_session );
    }
    
  }
}
