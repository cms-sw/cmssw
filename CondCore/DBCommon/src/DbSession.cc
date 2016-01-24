//local includes
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/BlobStreamerPluginFactory.h"
#include "CondCore/DBCommon/interface/Auth.h"
// CMSSW includes
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
#include "CondCore/ORA/interface/ConnectionPool.h"
// coral includes
#include "RelationalAccess/ISessionProxy.h"

namespace cond {

  inline std::auto_ptr<cond::TechnologyProxy> buildTechnologyProxy(const std::string&userconnect, 
								   const DbConnection& connection){
    std::string protocol;
    std::size_t pos=userconnect.find_first_of(':');
    if( pos!=std::string::npos ){
      protocol=userconnect.substr(0,pos);
      std::size_t p=protocol.find_first_of('_');
      if(p!=std::string::npos){
	protocol=protocol.substr(0,p);
      }
    }else{
      throw cond::Exception(userconnect +":connection string format error");
    }
    std::auto_ptr<cond::TechnologyProxy> ptr(cond::TechnologyProxyFactory::get()->create(protocol));
    (*ptr).initialize(connection);
    return ptr;
  }
  
  class SessionImpl {
    public:
      SessionImpl():
        connection(),
        blobStreamingService( "COND/Services/BlobStreamingService" ),
        database(),
        transaction(),
        isOpen(false){
      }

      explicit SessionImpl( const DbConnection& connection ):
        connection(new DbConnection(connection)),
        blobStreamingService( "COND/Services/BlobStreamingService" ),
        database(),
        transaction(),
        isOpen(false){
      }
      
    
      virtual ~SessionImpl(){
        close();
      }
    
    void open( const std::string& connectionString, 
	       const std::string& role,
	       bool readOnly ){
        close();
        if( connection.get() ){
          if(!connection->isOpen()){
            throw cond::Exception("DbSession::open: cannot open session. Underlying connection is closed.");
          }
          boost::shared_ptr<ora::ConnectionPool> connPool = connection->connectionPool();
          database.reset( new ora::Database( connPool ) );
 
          ora::IBlobStreamingService* blobStreamer = cond::BlobStreamerPluginFactory::get()->create(  blobStreamingService );
          if(!blobStreamer) throw cond::Exception("DbSession::open: cannot find required plugin. No instance of ora::IBlobStreamingService has been loaded..");
          database->configuration().setBlobStreamingService( blobStreamer );
          //database->configuration().properties().setFlag( ora::Configuration::automaticDatabaseCreation() );
          database->configuration().properties().setFlag( ora::Configuration::automaticContainerCreation() );
          // open the db connection
          technologyProxy = buildTechnologyProxy(connectionString, *connection);
          std::string connStr = (*technologyProxy).getRealConnectString( connectionString );
          database->connect( connStr, role, readOnly );
          transaction.reset( new cond::DbTransaction( database->transaction() ) );
          isOpen = true;
        }
      }

    void openReadOnly( const std::string& connectionString, 
		       const std::string& transactionId ){
        close();
        if( connection.get() ){
          if(!connection->isOpen()){
            throw cond::Exception("DbSession::open: cannot open session. Underlying connection is closed.");
          }
          boost::shared_ptr<ora::ConnectionPool> connPool = connection->connectionPool();
          database.reset( new ora::Database( connPool ) );
 
          ora::IBlobStreamingService* blobStreamer = cond::BlobStreamerPluginFactory::get()->create(  blobStreamingService );
          if(!blobStreamer) throw cond::Exception("DbSession::open: cannot find required plugin. No instance of ora::IBlobStreamingService has been loaded..");
          database->configuration().setBlobStreamingService( blobStreamer );
          // open the db connection
          technologyProxy = buildTechnologyProxy(connectionString, *connection);
          std::string connStr = (*technologyProxy).getRealConnectString(connectionString, transactionId);
          database->connect( connStr, Auth::COND_READER_ROLE, true );
          transaction.reset( new cond::DbTransaction( database->transaction() ) );
          isOpen = true;
        }
      }

      void open( boost::shared_ptr<coral::ISessionProxy>& coralSession, 
		 const std::string& connectionString, 
		 const std::string& schemaName ){
        close();
	database.reset( new ora::Database );
	
	ora::IBlobStreamingService* blobStreamer = cond::BlobStreamerPluginFactory::get()->create(  blobStreamingService );
	if(!blobStreamer) throw cond::Exception("DbSession::open: cannot find required plugin. No instance of ora::IBlobStreamingService has been loaded..");
	database->configuration().setBlobStreamingService( blobStreamer );
	database->configuration().properties().setFlag( ora::Configuration::automaticContainerCreation() );
	database->connect( coralSession, connectionString, schemaName );
	transaction.reset( new cond::DbTransaction( database->transaction(), false ) );
	isOpen = true;
      }

      void close(){
        transaction.reset();
        database.reset();
        isOpen = false;
      }

      std::auto_ptr<DbConnection> connection;
      std::auto_ptr<cond::TechnologyProxy> technologyProxy;
      std::string const blobStreamingService;
      std::auto_ptr<ora::Database> database;
      std::auto_ptr<DbTransaction> transaction;
      bool isOpen;
  };

}

const char* cond::DbSession::COND_SCHEMA_VERSION = "2.0.0";   
const char* cond::DbSession::CHANGE_SCHEMA_VERSION = "2.0.0";

cond::DbSession::DbSession():
  m_implementation( new SessionImpl ){ 
}

cond::DbSession::DbSession( const DbConnection& connection ):
  m_implementation( new SessionImpl ( connection ) ){
}

cond::DbSession::DbSession( const DbSession& rhs ):
  m_implementation( rhs.m_implementation ){ 
}

cond::DbSession::~DbSession(){
}

cond::DbSession& cond::DbSession::operator=( const cond::DbSession& rhs ){
  if(this!=&rhs) m_implementation = rhs.m_implementation;
  return *this;
}

void cond::DbSession::open( const std::string& connectionString, bool readOnly )
{
  std::string emptyRole("");
  m_implementation->open( connectionString, emptyRole, readOnly );
}

void cond::DbSession::open( const std::string& connectionString, const std::string& asRole, bool readOnly )
{
  m_implementation->open( connectionString, asRole, readOnly );
}

void cond::DbSession::openReadOnly( const std::string& connectionString, const std::string& id )
{
  m_implementation->openReadOnly( connectionString, id );
}

void cond::DbSession::open( boost::shared_ptr<coral::ISessionProxy>& coralSession, const std::string& connectionString, const std::string& schemaName ){
  m_implementation->open( coralSession, connectionString, schemaName  );
}

void cond::DbSession::close()
{
  m_implementation->close();
}

bool cond::DbSession::isOpen() const {
  return m_implementation->isOpen;
}

const std::string& cond::DbSession::connectionString() const {
  if(!m_implementation->database.get())
    throw cond::Exception("DbSession::connectionString: cannot get connection string. Session has not been open.");
  return m_implementation->database->connectionString();
}

cond::DbConnection const & cond::DbSession::connection() const {
  return *(m_implementation->connection);
}


bool cond::DbSession::isTransactional() const {
  return m_implementation->technologyProxy->isTransactional();
}

const std::string& cond::DbSession::blobStreamingService() const 
{
  return m_implementation->blobStreamingService;
}

cond::DbTransaction& cond::DbSession::transaction()
{
  if(!m_implementation->transaction.get())
    throw cond::Exception("DbSession::transaction: cannot get transaction. Session has not been open.");
  return *m_implementation->transaction;
}

ora::Database& cond::DbSession::storage(){
  if(!m_implementation->database.get())
    throw cond::Exception("DbSession::storage: cannot access the database. Session has not been open.");
  return *m_implementation->database;
}

bool cond::DbSession::createDatabase(){
  bool created = false;
  if ( !storage().exists() ){
    created = true;
    storage().create( std::string(COND_SCHEMA_VERSION) );
  }  
  return created;
}

bool cond::DbSession::isOldSchema()
{
  ora::Version dbVer = storage().schemaVersion();
  if (dbVer == ora::Version::poolSchemaVersion()) return true;
  dbVer = storage().schemaVersion( true );
  return dbVer < ora::Version::fromString( std::string( CHANGE_SCHEMA_VERSION ) );
}

coral::ISchema& cond::DbSession::schema( const std::string& schemaName )
{
  return storage().storageAccessSession().get().schema( schemaName );
}

coral::ISchema& cond::DbSession::nominalSchema()
{
  return storage().storageAccessSession().get().nominalSchema(); 
}

bool cond::DbSession::deleteMapping( const std::string& mappingVersion ){
  ora::DatabaseUtility utility = storage().utility();
  utility.eraseMapping( mappingVersion );
  return true;
}

bool cond::DbSession::importMapping( const std::string& sourceConnectionString,
                                     const std::string& contName ){ 
  ora::DatabaseUtility utility = storage().utility();
  std::auto_ptr<cond::TechnologyProxy> technologyProxy = buildTechnologyProxy(sourceConnectionString, *(m_implementation->connection));
  utility.importContainerSchema( (*technologyProxy).getRealConnectString( sourceConnectionString ), contName );
  return true;
}

std::string cond::DbSession::storeObject( const ora::Object& object, const std::string& containerName  ){
  ora::OId oid = storage().insertItem( containerName, object );
  storage().flush();
  return oid.toString();
}

ora::Object  cond::DbSession::getObject( const std::string& objectId ){
  ora::OId oid;
  oid.fromString( objectId );
  return storage().fetchItem( oid );
}

bool cond::DbSession::deleteObject( const std::string& objectId ){
  ora::OId oid;
  oid.fromString( objectId );
  storage().erase( oid );
  storage().flush();
  return true;
}

std::string cond::DbSession::importObject( cond::DbSession& fromDatabase, const std::string& objectId ){
  ora::OId oid;
  oid.fromString( objectId );
  ora::Object data = fromDatabase.getObject( objectId );
  ora::Container cont = fromDatabase.storage().containerHandle( oid.containerId() );
  std::string ret = storeObject( data, cont.name() );
  data.destruct();
  return ret;
}

std::string cond::DbSession::classNameForItem( const std::string& objectId ){
  ora::OId oid;
  oid.fromString( objectId );
  std::string ret("");
  if( !oid.isInvalid() ){
    ora::Container cont = storage().containerHandle( oid.containerId() );
    ret = cont.realClassName();
  }
  return ret; 
}

void cond::DbSession::flush(){
  storage().flush();
}

cond::PoolTokenParser::PoolTokenParser( ora::Database& db ):
  m_db( db ){
}

ora::OId cond::PoolTokenParser::parse( const std::string& poolToken ){
  std::pair<std::string,int> oidData = parseToken( poolToken );
  if( oidData.first.empty() ){
    throwException("Could not resolve Container name from token=\""+poolToken+"\".","PoolTokenParser::parse");
  }
  ora::Container cont = m_db.containerHandle(  oidData.first );
  return ora::OId( cont.id(), oidData.second );
}

std::string cond::PoolTokenParser::className( const std::string& oraToken ){
  ora::OId oid;
  oid.fromString( oraToken );
  ora::Container cont = m_db.containerHandle(  oid.containerId() );
  return cont.className();
}

cond::PoolTokenWriter::PoolTokenWriter( ora::Database& db ):
  m_db( db ){
}

std::string cond::PoolTokenWriter::write( const ora::OId& oid ){
  ora::Container cont = m_db.containerHandle( oid.containerId() );
  return writeToken( cont.name(), oid.containerId(), oid.itemId(), cont.className() );
}
