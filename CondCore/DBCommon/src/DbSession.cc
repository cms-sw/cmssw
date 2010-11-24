//local includes
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/BlobStreamerPluginFactory.h"
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
    //std::cout<<"userconnect "<<userconnect<<std::endl;
    //std::cout<<"protocol "<<protocol<<std::endl;  
    std::auto_ptr<cond::TechnologyProxy> ptr(cond::TechnologyProxyFactory::get()->create(protocol));
    (*ptr).initialize(userconnect,connection);
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
          database->configuration().properties().setFlag( ora::Configuration::automaticDatabaseCreation() );
          // open the db connection
          technologyProxy = buildTechnologyProxy(connectionString, *connection);
          std::string connStr = (*technologyProxy).getRealConnectString();
          database->connect( connStr, readOnly );
          transaction.reset( new cond::DbTransaction( database->transaction() ) );
          isOpen = true;
        }
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
  m_implementation->open( connectionString, readOnly );
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
  if(!m_implementation->connection.get() || !m_implementation->connection->isOpen())
    throw cond::Exception("DbSession::transaction: cannot open transaction. Underlying connection is closed.");
  if(!m_implementation->transaction.get())
    throw cond::Exception("DbSession::transaction: cannot get transaction. Session has not been open.");
  return *m_implementation->transaction;
}

ora::Database& cond::DbSession::storage(){
  if(!m_implementation->connection.get() || !m_implementation->connection->isOpen())
    throw cond::Exception("DbSession::storage: cannot access the storage. Underlying connection is closed.");
  if(!m_implementation->database.get())
    throw cond::Exception("DbSession::storage: cannot access the database. Session has not been open.");
  return *m_implementation->database;
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
  utility.importContainerSchema( (*technologyProxy).getRealConnectString(), contName );
  return true;
}

std::string cond::DbSession::storeObject( const ora::Object& object, const std::string& containerName  ){
  ora::OId oid = storage().insertItem( containerName, object );
  storage().flush();
  int oid0 = oid.containerId(); // no clue why in POOL contId does not start from 0...
  return writeToken( containerName, oid0, oid.itemId(), object.typeName() );
}

ora::Object  cond::DbSession::getObject( const std::string& objectId ){
  std::pair<std::string,int> oidData = parseToken( objectId );
  ora::Container cont = storage().containerHandle(  oidData.first );
  return cont.fetchItem( oidData.second );
}

bool cond::DbSession::deleteObject( const std::string& objectId ){
  std::pair<std::string,int> oidData = parseToken( objectId );
  ora::Container cont = storage().containerHandle(  oidData.first );
  cont.erase( oidData.second );
  cont.flush();
  return true;
}

std::string cond::DbSession::importObject( cond::DbSession& fromDatabase, const std::string& objectId ){
  std::pair<std::string,int> oidData = parseToken( objectId );
  ora::Object data = fromDatabase.getObject( objectId );
  std::string tok = storeObject( data, oidData.first );
  data.destruct();
  return tok;
}

void cond::DbSession::flush(){
  storage().flush();
}

