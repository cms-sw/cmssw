#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Monitoring.h"
#include "DatabaseSession.h"
#include "IDatabaseSchema.h"
#include "Sequences.h"
#include "MappingDatabase.h"
#include "TransactionCache.h"
#include "DatabaseContainer.h"
#include "ClassUtils.h"
#include "MappingRules.h"
#include "DatabaseUtilitySession.h"
// externals
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"

ora::ContainerUpdateTable::ContainerUpdateTable():
  m_table(){
}

ora::ContainerUpdateTable::~ContainerUpdateTable(){
}

void ora::ContainerUpdateTable::takeNote( int contId,
                                          unsigned int size ){
  std::map<int, unsigned int>::iterator iC = m_table.find( contId );
  if( iC == m_table.end() ){
    iC = m_table.insert( std::make_pair( contId, 0 ) ).first;
  }
  iC->second = size;
}

void ora::ContainerUpdateTable::remove(  int contId ){
  m_table.erase( contId );
}

const std::map<int, unsigned int>& ora::ContainerUpdateTable::table(){
  return m_table;
}

void ora::ContainerUpdateTable::clear(){
  m_table.clear();
}

ora::DatabaseSession::DatabaseSession():
  m_connectionPool( new ConnectionPool ),
  m_dbSession(),
  m_connectionString( "" ),
  m_schema(),
  m_contIdSequence(),
  m_mappingDb(),
  m_transactionCache(),
  m_containerUpdateTable(),
  m_configuration(),
  m_monitoring(0){
  // for the private connection pool does not make sense to have a real pool... 
  m_connectionPool->configuration().setConnectionTimeOut(0);
}

ora::DatabaseSession::DatabaseSession(boost::shared_ptr<ConnectionPool>& connectionPool ):
  m_connectionPool( connectionPool ),
  m_dbSession(),
  m_connectionString( "" ),
  m_schema(),
  m_contIdSequence(),
  m_mappingDb(),
  m_transactionCache(),
  m_containerUpdateTable(),
  m_configuration(),
  m_monitoring(0){
}

ora::DatabaseSession::~DatabaseSession(){
  disconnect();
}

bool ora::DatabaseSession::connect( const std::string& connectionString,
                                    bool readOnly ){
  m_dbSession = m_connectionPool->connect( connectionString, readOnly?coral::ReadOnly:coral::Update );
  if(m_dbSession.isValid()) {
    m_connectionString = connectionString;
    if( ora::Monitoring::isEnabled() ){
      m_monitoring = ora::Monitoring::get().startSession( connectionString );
    }
  }
  return isConnected();
}

void ora::DatabaseSession::clearTransaction(){
  m_transactionCache.reset();
  m_mappingDb.reset();
  m_contIdSequence.reset();
  m_schema.reset();
  m_containerUpdateTable.clear();
}

void ora::DatabaseSession::disconnect(){
  if( isConnected() ){
    if( isTransactionActive()) rollbackTransaction();
  }
  clearTransaction();
  m_dbSession.close();
  m_connectionString.clear();
  if(m_monitoring) m_monitoring->stop();
}

bool ora::DatabaseSession::isConnected(){
  return m_dbSession.isValid();
}

const std::string& ora::DatabaseSession::connectionString(){
  return m_connectionString;
}

void ora::DatabaseSession::startTransaction( bool readOnly ){
  if( !m_transactionCache.get() ){
    m_dbSession.get().transaction().start( readOnly );
    m_schema.reset( IDatabaseSchema::createSchemaHandle( m_dbSession.get().nominalSchema() ));
    m_contIdSequence.reset( new NamedSequence( MappingRules::sequenceNameForContainerId(), *m_schema ));
    m_mappingDb.reset( new MappingDatabase( *m_schema ));
    m_transactionCache.reset( new TransactionCache );
    if(m_monitoring) {
      m_monitoring->newTransaction();
    }
  }
}

void ora::DatabaseSession::commitTransaction(){
  if( m_transactionCache.get() ){
    m_schema->containerHeaderTable().updateNumberOfObjects( m_containerUpdateTable.table() );
    m_dbSession.get().transaction().commit();
    clearTransaction();
    if(m_monitoring) {
      m_monitoring->stopTransaction();
    }
  }
}

void ora::DatabaseSession::rollbackTransaction(){
  if( m_transactionCache.get() ){
    m_dbSession.get().transaction().rollback();
    clearTransaction();
    if(m_monitoring) {
      m_monitoring->stopTransaction(false);
    }
  }
}

bool ora::DatabaseSession::isTransactionActive( bool checkIfReadOnly ){
  bool ret = false;
  if( m_dbSession.get().transaction().isActive() ){
    if( checkIfReadOnly ){
      if( m_dbSession.get().transaction().isReadOnly() ) ret = true;
    } else {
      ret = true;
    }
  }
  return ret;
}

bool ora::DatabaseSession::exists(){
  if(!m_transactionCache->dbExistsLoaded()){
    m_transactionCache->setDbExists( m_schema->exists() );
  }
  return m_transactionCache->dbExists();
}

void ora::DatabaseSession::create( const std::string& userSchemaVersion ){
  m_schema->create( userSchemaVersion );
  m_contIdSequence->create();
  m_mappingDb->setUp();
  m_transactionCache->setDbExists( true );
}

void ora::DatabaseSession::drop(){
  if(!testDropPermission()){
    throwException("Drop permission has been denied for the current user.",
		   "DatabaseSession::drop");
  }
  m_schema->drop();
  m_transactionCache->dropDatabase();
}

void ora::DatabaseSession::setAccessPermission( const std::string& principal, 
						bool forWrite ){
  m_schema->setAccessPermission( principal, forWrite );
}
    
bool ora::DatabaseSession::testDropPermission(){
  if(!m_transactionCache->dropPermissionLoaded()){
    m_transactionCache->setDropPermission( m_schema->testDropPermission() );
  }
  return m_transactionCache->dropPermission();
}

void ora::DatabaseSession::open(){
  if(!m_transactionCache->isLoaded()){
    std::map<std::string, ContainerHeaderData> containersData;
    m_schema->containerHeaderTable().getContainerData( containersData );
    for(std::map<std::string, ContainerHeaderData>::iterator iC = containersData.begin();
        iC != containersData.end(); ++iC){
      Handle<DatabaseContainer> container( new DatabaseContainer( iC->second.id, iC->first,
                                                                  iC->second.className,
                                                                  iC->second.numberOfObjects, *this ) );
      m_transactionCache->addContainer( iC->second.id, iC->first, container );
    }
    m_schema->mainTable().getParameters( m_transactionCache->dbParams() );
    m_transactionCache->setLoaded();
  }
}

std::string ora::DatabaseSession::schemaVersion( bool userSchema ){
  std::map<std::string,std::string>& params = m_transactionCache->dbParams();
  std::string version("");
  std::string paramName = IMainTable::versionParameterName();
  if(userSchema ) paramName = IMainTable::userSchemaVersionParameterName();
  std::map<std::string,std::string>::const_iterator iPar = params.find( paramName );
  if( iPar != params.end() ){
    version = iPar->second;
  }
  return version;
}

ora::Handle<ora::DatabaseContainer> ora::DatabaseSession::addContainer( const std::string& containerName,
                                                                        const std::string& className ){
  int newContId = m_contIdSequence->getNextId( true );
  m_schema->containerHeaderTable().addContainer( newContId, containerName, className );
  Handle<DatabaseContainer> container( new DatabaseContainer( newContId, containerName,
                                                              className, 0, *this ) );
  m_transactionCache->addContainer( newContId, containerName, container );
  return container;
}


ora::Handle<ora::DatabaseContainer> ora::DatabaseSession::createContainer( const std::string& containerName,
                                                                           const Reflex::Type& type ){
  // create the container
  int newContId = m_contIdSequence->getNextId( true );
  Handle<DatabaseContainer> newCont ( new DatabaseContainer( newContId, containerName, type, *this ) );
  m_transactionCache->addContainer( newContId, containerName, newCont );
  newCont->create();
  return newCont;
}

void ora::DatabaseSession::dropContainer( const std::string& name ){
  Handle<DatabaseContainer> cont = m_transactionCache->getContainer( name );
  m_transactionCache->eraseContainer( cont->id(), name );
  cont->drop();
}

ora::Handle<ora::DatabaseContainer> ora::DatabaseSession::containerHandle( const std::string& name ){
  return m_transactionCache->getContainer( name );
}

ora::Handle<ora::DatabaseContainer> ora::DatabaseSession::containerHandle( int contId ){
  return  m_transactionCache->getContainer( contId );
}

const std::map<int, ora::Handle<ora::DatabaseContainer> >& ora::DatabaseSession::containers(){
  return m_transactionCache->containers();
}

void ora::DatabaseSession::setObjectName( const std::string& name, 
                                          int containerId, 
                                          int itemId ){
  m_schema->namingServiceTable().setObjectName( name, containerId, itemId );
}

bool ora::DatabaseSession::eraseObjectName( const std::string& name ){
  return m_schema->namingServiceTable().eraseObjectName( name );
}

bool ora::DatabaseSession::eraseAllNames(){
  return m_schema->namingServiceTable().eraseAllNames();
}

ora::Object ora::DatabaseSession::fetchObjectByName( const std::string& name ){
  ora::Object ret;
  std::pair<int,int> oid;
  if( m_schema->namingServiceTable().getObjectByName( name, oid ) ){
    ora::Handle<ora::DatabaseContainer> cont = containerHandle( oid.first );
    if( cont ) ret = Object( cont->fetchItem( oid.second ), cont->type() );
  }
  return ret;
}

bool ora::DatabaseSession::getItemId( const std::string& name, ora::OId& destination ){
  std::pair<int,int> oidData;
  if( m_schema->namingServiceTable().getObjectByName( name, oidData ) ){
    destination = OId( oidData.first, oidData.second );
    return true;
  }
  return false;
}

boost::shared_ptr<void> ora::DatabaseSession::fetchTypedObjectByName( const std::string& name, 
                                                                      const Reflex::Type& asType ){
  boost::shared_ptr<void> ret = m_transactionCache->getNamedReference( name );
  if( !ret.get() ){
    std::pair<int,int> oid;
    if( m_schema->namingServiceTable().getObjectByName( name, oid ) ){
      ora::Handle<ora::DatabaseContainer> cont = containerHandle( oid.first );
      void* ptr = 0;
      if( cont ) {
        ptr = cont->fetchItemAsType( oid.second, asType );
        if( ptr) ret = boost::shared_ptr<void>( ptr, RflxDeleter( cont->type() ) );
      }
    }
    if( ret.get() ) m_transactionCache->setNamedReference( name, ret );
  }
  return ret;
}

bool ora::DatabaseSession::getNamesForContainer( int containerId, 
                                                 std::vector<std::string>& destination ){
  return m_schema->namingServiceTable().getNamesForContainer( containerId, destination );
}
    
bool ora::DatabaseSession::getNamesForObject( int containerId, 
                                              int itemId, 
                                              std::vector<std::string>& destination ){
  return m_schema->namingServiceTable().getNamesForObject( containerId, itemId, destination );
}

bool ora::DatabaseSession::listObjectNames( std::vector<std::string>& destination ){
  
  return m_schema->namingServiceTable().getAllNames( destination );
}

ora::Handle<ora::DatabaseUtilitySession> ora::DatabaseSession::utility(){
  if( !m_transactionCache->utility() ){
    Handle<DatabaseUtilitySession> util ( new DatabaseUtilitySession( *this ) );
    m_transactionCache->setUtility( util );
  }
  return m_transactionCache->utility();
}

ora::IDatabaseSchema& ora::DatabaseSession::schema(){
  return *m_schema;  
}

ora::NamedSequence& ora::DatabaseSession::containerIdSequence(){
  return *m_contIdSequence;
}

ora::MappingDatabase& ora::DatabaseSession::mappingDatabase(){
  return *m_mappingDb;  
}

ora::Configuration& ora::DatabaseSession::configuration(){
  return m_configuration;
}

ora::SharedSession& ora::DatabaseSession::storageAccessSession(){
  return m_dbSession;
}

boost::shared_ptr<ora::ConnectionPool>& ora::DatabaseSession::connectionPool(){
  return m_connectionPool;
}

ora::ContainerUpdateTable& ora::DatabaseSession::containerUpdateTable(){
  return m_containerUpdateTable;
}
