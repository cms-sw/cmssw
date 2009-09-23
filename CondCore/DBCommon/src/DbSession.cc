//local includes
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ClassInfoLoader.h"
#include "CondCore/DBCommon/interface/BlobStreamerPluginFactory.h"
// CMSSW includes
#include "FWCore/PluginManager/interface/PluginFactory.h"
// coral includes
#include "RelationalAccess/ISessionProxy.h"
// pool includes
#include "POOLCore/Token.h"
#include "FileCatalog/IFileCatalog.h"
#include "DataSvc/IDataSvc.h"
#include "DataSvc/DataSvcFactory.h"
#include "PersistencySvc/IConfiguration.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "StorageSvc/DbType.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingUtilities.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingSchema.h"
//#include "ObjectRelationalAccess/ObjectRelationalMappingPersistency.h"

namespace cond {
  class DbSession::SessionImpl {
    public:
      SessionImpl();
      explicit SessionImpl( const DbConnection& connection );

      virtual ~SessionImpl();

      void open( const std::string& connectionString, bool readOnly );
      void close();

      DbConnection* m_connection;
      std::string m_connectionString;
      std::string m_blobStreamingService;
      pool::IFileCatalog* m_catalogue;
      pool::IDataSvc* m_dataSvc;
      coral::ISessionProxy* m_session;
      DbTransaction* m_transaction;
      bool m_isOpen;
  };
  
}


cond::DbSession::SessionImpl::SessionImpl():
  m_connection(0),
  m_connectionString(""),
  m_blobStreamingService( "" ),
  m_catalogue(0),
  m_dataSvc(0),
  m_session(0),
  m_transaction(0),
  m_isOpen(false){
}

cond::DbSession::SessionImpl::SessionImpl( const DbConnection& connection ):
  m_connection(new DbConnection(connection)),
  m_connectionString(""),
  m_blobStreamingService( "" ),
  m_catalogue(0),
  m_dataSvc(0),
  m_session(0),
  m_transaction(0),
  m_isOpen(false){
}

cond::DbSession::SessionImpl::~SessionImpl(){
  close();
  if(m_connection) delete m_connection;
}

void cond::DbSession::SessionImpl::open( const std::string& connectionString, bool readOnly ){
  close();
  if(m_connection){
    if(!m_connection->isOpen())
      throw cond::Exception("DbSession::open: cannot open session. Underlying connection is closed.");
    
    m_catalogue = new pool::IFileCatalog();
    m_dataSvc = pool::DataSvcFactory::instance(m_catalogue);
    // pool configuration
    m_dataSvc->configuration().setConnectionService( &m_connection->connectionService(), false );
    m_dataSvc->configuration().enableSessionSharing();
    std::string pluginName("COND/Services/TBufferBlobStreamingService2");
    if(!m_blobStreamingService.empty()){
      pluginName = m_blobStreamingService;
    }
    pool::IBlobStreamingService* blobStreamer = cond::BlobStreamerPluginFactory::get()->create( pluginName, pluginName);
    if(!blobStreamer) throw cond::Exception("DbSession::open: cannot find required plugin. No instance of pool::IBlobStreamingService has been loaded..");
    m_dataSvc->configuration().setBlobStreamer( blobStreamer, false );
    pool::DatabaseConnectionPolicy policy;
    policy.setWriteModeForNonExisting( pool::DatabaseConnectionPolicy::CREATE );
    policy.setWriteModeForExisting( pool::DatabaseConnectionPolicy::UPDATE );
    policy.setReadMode( pool::DatabaseConnectionPolicy::READ );
    m_dataSvc->session().setDefaultConnectionPolicy( policy );
    // open the db connection
    m_connectionString = connectionString;
    m_session = &m_dataSvc->configuration().sharedSession( connectionString, (readOnly)? coral::ReadOnly: coral::Update );
    std::string catalogConnectionString("pfncatalog_memory://POOL_RDBMS?");
    catalogConnectionString.append(connectionString);
    m_catalogue->setWriteCatalog( catalogConnectionString );
    m_catalogue->connect();
    m_catalogue->start();
    m_transaction = new cond::DbTransaction( *m_session, m_dataSvc->session() );
    m_isOpen = true;
  }
  
}

void cond::DbSession::SessionImpl::close(){
  m_connectionString.clear();
  if(m_session){
    m_session = 0;
  }
  if( m_transaction ) {
    delete m_transaction;
    m_transaction = 0;
  }
  if(m_catalogue) {
    m_catalogue->commit();
    m_catalogue->disconnect();
    delete m_catalogue;
    m_catalogue = 0;
  }
  if( m_dataSvc ) {
    m_dataSvc->session().disconnectAll();
    delete m_dataSvc;
    m_dataSvc = 0;
  }
  m_isOpen = false;
}

cond::DbSession::DbSession():
  m_implementation(new SessionImpl()){ 
}

cond::DbSession::DbSession( const DbConnection& connection ):
  m_implementation(new SessionImpl(connection)){ 
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
  return m_implementation->m_isOpen;
}

const std::string& cond::DbSession::connectionString() const {
  return m_implementation->m_connectionString;
}

void cond::DbSession::setBlobStreamingService( const std::string& serviceName )
{
  m_implementation->m_blobStreamingService = serviceName;
}

const std::string& cond::DbSession::blobStreamingService() const 
{
  return m_implementation->m_blobStreamingService;
}

cond::DbTransaction& cond::DbSession::transaction()
{
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::transaction: cannot get transaction. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::transaction: cannot open transaction. Underlying connection is closed.");
  return *m_implementation->m_transaction;
}

coral::ISchema& cond::DbSession::schema( const std::string& schemaName )
{
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::schema: cannot get schema. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::schema: cannot get schema. Underlying connection is closed.");
  return m_implementation->m_session->schema( schemaName );  
}

coral::ISchema& cond::DbSession::nominalSchema()
{
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::nominalSchema: cannot get schema. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::nominalSchema: cannot get schema. Underlying connection is closed.");
  return m_implementation->m_session->nominalSchema();  
}

coral::ISessionProxy& cond::DbSession::coralSession()
{
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::coralSession: cannot get coral session. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::coralSession: cannot get coral session. Underlying connection is closed.");
  return *m_implementation->m_session;  
}

pool::IDataSvc& cond::DbSession::poolCache()
{
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::poolCache: cannot get pool cache. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::poolCache: cannot get pool cache. Underlying connection is closed.");
  return *m_implementation->m_dataSvc;
}

bool cond::DbSession::initializeMapping(const std::string& mappingVersion, const std::string& xmlStream){
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::initializeMapping: cannot get coral session. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::initializeMapping: cannot get coral session. Underlying connection is closed.");
  bool ret = false;
  pool::ObjectRelationalMappingSchema mappingSchema( m_implementation->m_session->nominalSchema() );
  pool::ObjectRelationalMappingUtilities mappingUtil( m_implementation->m_session );
  if( !mappingSchema.existTables() || !mappingUtil.existsMapping(mappingVersion) ){
    mappingUtil.buildAndMaterializeMappingFromBuffer( xmlStream.c_str(),false,false );
    ret = true;
  }
  return ret;
}

bool cond::DbSession::deleteMapping( const std::string& mappingVersion, bool removeTables ){
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::deleteMapping: cannot get coral session. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::deleteMapping: cannot get coral session. Underlying connection is closed.");
  bool ret = false;
  pool::ObjectRelationalMappingSchema mappingSchema( m_implementation->m_session->nominalSchema() );
  pool::ObjectRelationalMappingUtilities mappingUtil( m_implementation->m_session );
  if( mappingSchema.existTables() && mappingUtil.existsMapping( mappingVersion ) ){
    mappingUtil.removeMapping(mappingVersion,removeTables);
    ret = true;
  }
  return ret;
}

bool cond::DbSession::importMapping( cond::DbSession& fromDatabase,
                                     const std::string& contName,
                                     const std::string& classVersion,
                                     bool allVersions ){
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::importMapping: cannot get coral session. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::importMapping: cannot get coral session. Underlying connection is closed.");
  if(!fromDatabase.m_implementation->m_session)
    throw cond::Exception("DbSession::importMapping: cannot get source coral session. Session has not been open.");
  if(!fromDatabase.m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::importMapping: cannot get source coral session. Underlying connection is closed.");
  
  pool::ObjectRelationalMappingUtilities mappingutil( fromDatabase.m_implementation->m_session );
  mappingutil.loadMappingInformation( contName, classVersion, allVersions);
  mappingutil.setSession( m_implementation->m_session,false);
  return mappingutil.storeMappingInformation(true);  
}



bool cond::DbSession::storeObject( pool::RefBase& objectRef, const std::string& containerName  ){
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::storeObject: cannot access object store. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::storeObject: cannot access object store. Underlying connection is closed.");
  pool::Placement place;
  place.setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  place.setDatabase(m_implementation->m_connectionString,
                    pool::DatabaseSpecification::PFN);
  place.setContainerName(containerName);
  return objectRef.markWrite(place);
}

pool::RefBase cond::DbSession::getObject( const std::string& objectId ){
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::getObject: cannot access object store. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::getObject: cannot access object store. Underlying connection is closed.");
  pool::RefBase theObject(m_implementation->m_dataSvc, objectId, cond::reflexTypeByToken(objectId).TypeInfo());
  return theObject;
}

bool cond::DbSession::deleteObject( const std::string& objectId ){
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::deleteObject: cannot access object store. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::deleteObject: cannot access object store. Underlying connection is closed.");
  pool::RefBase objectRef = getObject( objectId );
  return objectRef.markDelete();
}

std::string cond::DbSession::importObject( cond::DbSession& fromDatabase, const std::string& objectId ){
  if(!m_implementation->m_session)
    throw cond::Exception("DbSession::importObject: cannot access destination object store. Session has not been open.");
  if(!m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::importObject: cannot access destination object store. Underlying connection is closed.");
  if(!fromDatabase.m_implementation->m_session)
    throw cond::Exception("DbSession::importObject: cannot access source object store. Session has not been open.");
  if(!fromDatabase.m_implementation->m_connection->isOpen())
    throw cond::Exception("DbSession::importObject: cannot access source object store. Underlying connection is closed.");
  pool::RefBase source = fromDatabase.getObject( objectId );
  pool::RefBase dest( m_implementation->m_dataSvc,
                      source.object().get(),
                      source.objectType().TypeInfo() );
  pool::Placement destPlace;
  destPlace.setDatabase(m_implementation->m_connectionString,
                        pool::DatabaseSpecification::PFN );
  destPlace.setContainerName(source.token()->contID());
  destPlace.setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  dest.markWrite( destPlace );
  return dest.toString();
}
