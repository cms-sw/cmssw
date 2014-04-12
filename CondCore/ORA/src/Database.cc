#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Handle.h"
#include "DatabaseSession.h"
#include "DatabaseContainer.h"
#include "TransactionCache.h"
#include "ContainerSchema.h"
#include "IDatabaseSchema.h"
#include "ClassUtils.h"

namespace ora {

  class DatabaseImpl {
    public:
      DatabaseImpl():
        m_session(0),
        m_transaction(0){
        m_session.reset( new DatabaseSession );
        m_transaction.reset( new Transaction( *m_session ));
      }

      DatabaseImpl(boost::shared_ptr<ConnectionPool>& connectionPool):
        m_session(0),
        m_transaction(0){
        m_session.reset( new DatabaseSession( connectionPool ) );
        m_transaction.reset( new Transaction( *m_session )) ;
      }
      
      ~DatabaseImpl(){
      }

      std::auto_ptr<DatabaseSession> m_session;
      std::auto_ptr<Transaction> m_transaction;      
  };
  
  std::string nameFromClass( const Reflex::Type& contType ){
    return contType.Name( Reflex::SCOPED );
  }
  
  Container getContainerFromSession( const std::string& name, const Reflex::Type& contType, DatabaseSession& session ){
    Handle<DatabaseContainer> contHandle = session.containerHandle( name );
    if( !contHandle ){
      if( session.configuration().properties().getFlag( Configuration::automaticDatabaseCreation()) ||
          session.configuration().properties().getFlag( Configuration::automaticContainerCreation() ) ){
        contHandle = session.createContainer( name, contType );
      } else {
        throwException("Container \""+name+"\" does not exist in the database.",
                       "Database::getContainerFromSession");
      }
    }

    return Container( contHandle );
  }
}

std::string ora::Database::nameForContainer( const std::type_info& typeInfo ){
  Reflex::Type contType = ClassUtils::lookupDictionary( typeInfo );
  return nameFromClass( contType );
}

std::string ora::Database::nameForContainer( const std::string& className ){
  return className;
}

ora::Database::Database():
  m_impl( new DatabaseImpl ){
}

ora::Database::Database( const Database& rhs ):
  m_impl( rhs.m_impl ){
}

ora::Database::Database(boost::shared_ptr<ConnectionPool>& connectionPool):
  m_impl( new DatabaseImpl( connectionPool) ){
}

ora::Database::~Database(){
}

ora::Database& ora::Database::operator=( const Database& rhs ){
  if( this != &rhs ) m_impl = rhs.m_impl;
  return *this;
}

ora::Configuration& ora::Database::configuration(){
  return m_impl->m_session->configuration();
}

bool ora::Database::connect( const std::string& connectionString,
                             bool readOnly ){
  return m_impl->m_session->connect( connectionString, readOnly );
}

bool ora::Database::connect( const std::string& connectionString,
			     const std::string& asRole,
                             bool readOnly ){
  return m_impl->m_session->connect( connectionString, asRole, readOnly );
}

bool ora::Database::connect( boost::shared_ptr<coral::ISessionProxy>& coralSession, const std::string& connectionString, const std::string& schemaName ){
  return m_impl->m_session->connect( coralSession, connectionString, schemaName );
}

void ora::Database::disconnect(){
  m_impl->m_session->disconnect();
}

bool ora::Database::isConnected() {
  return m_impl->m_session->isConnected();
}

const std::string& ora::Database::connectionString() {
  return m_impl->m_session->connectionString();
}

ora::Transaction& ora::Database::transaction(){
  if(!m_impl->m_session->isConnected()) {
    throwException("No database storage connected.","Database::transaction");
  }
  return *m_impl->m_transaction;
}

void ora::Database::checkTransaction(){
  if(!m_impl->m_session->isConnected()) {
    throwException("No database storage connected.","Database::checkTransaction");
  }
  if(!m_impl->m_transaction->isActive()) {
    throwException("Transaction is not active.","Database::checkTransaction");
  }  
}

bool ora::Database::exists(){
  checkTransaction();
  return m_impl->m_session->exists();
}

bool ora::Database::create( std::string userSchemaVersion ){
  bool created = false;
  if( !exists()){
    m_impl->m_session->create( userSchemaVersion );
    created = true;
  }
  return created;
}

bool ora::Database::drop(){
  bool dropped = false;
  if( exists()){
    open();
    const std::map<int, Handle<DatabaseContainer> >& conts = m_impl->m_session->containers();
    for(std::map<int, Handle<DatabaseContainer> >::const_iterator iC = conts.begin();
        iC != conts.end(); iC++ ){
      iC->second->drop();
    }
    m_impl->m_session->drop();
    dropped = true;
  }
  return dropped;
}

void ora::Database::setAccessPermission( const std::string& principal, bool forWrite ){
  if( exists()){
    open();
    m_impl->m_session->setAccessPermission( principal, forWrite );
    const std::map<int, Handle<DatabaseContainer> >& conts = m_impl->m_session->containers();
    for(std::map<int, Handle<DatabaseContainer> >::const_iterator iC = conts.begin();
        iC != conts.end(); iC++ ){
      iC->second->setAccessPermission( principal, forWrite );
    }
  }
}

void ora::Database::open( bool writingAccess /*=false*/){
  checkTransaction();
  if( !m_impl->m_session->exists() ){
    if( writingAccess && m_impl->m_session->configuration().properties().getFlag( Configuration::automaticDatabaseCreation() ) ){
      m_impl->m_session->create();
    } else {
      throwException("Database does not exists in \""+m_impl->m_session->connectionString()+"\"","Database::open");
    }
  }
  m_impl->m_session->open();
}

ora::Version ora::Database::schemaVersion( bool userSchema ){
  checkTransaction();
  if( !m_impl->m_session->exists() ){
    throwException("Database does not exists in \""+m_impl->m_session->connectionString()+"\"","Database::schemaVersion");
  }
  return Version::fromString( m_impl->m_session->schemaVersion( userSchema ) );
}

std::set< std::string > ora::Database::containers() {
  open();
  std::set< std::string > contList;
  const std::map<int, Handle<DatabaseContainer> >& conts = m_impl->m_session->containers();
  for(std::map<int, Handle<DatabaseContainer> >::const_iterator iC = conts.begin();
      iC != conts.end(); iC++ ){
    contList.insert( iC->second->name() );
  }
  return contList;
}

ora::Container ora::Database::createContainer( const std::string& name,
                                               const std::type_info& typeInfo ){
  open( true );
  if( m_impl->m_session->containerHandle( name ) ){
    throwException("Container with name \""+name+"\" already exists in the database.",
                   "Database::createContainer");
  }
  Reflex::Type contType = ClassUtils::lookupDictionary( typeInfo );
  Handle<DatabaseContainer> cont = m_impl->m_session->createContainer( name, contType );
  return Container( cont );
}

ora::Container ora::Database::createContainer( const std::type_info& typeInfo ){
  open( true );
  Reflex::Type contType = ClassUtils::lookupDictionary( typeInfo );
  std::string name = nameFromClass( contType );
  if( m_impl->m_session->containerHandle( name ) ){
    throwException("Container with name \""+name+"\" already exists in the database.",
                   "Database::createContainer");
  }  
  Handle<DatabaseContainer> cont = m_impl->m_session->createContainer( name, contType );
  return Container( cont );
}

ora::Container ora::Database::createContainer( const std::string& className,
                                               std::string name ){
  open( true );
  Reflex::Type contType =  ClassUtils::lookupDictionary( className );
  if( name.empty() ) name = nameForContainer( className );
  if( m_impl->m_session->containerHandle( name ) ){
    throwException("Container with name \""+name+"\" already exists in the database.",
                   "Database::createContainer");
  }  
  Handle<DatabaseContainer> cont = m_impl->m_session->createContainer( name, contType );
  return Container( cont );
}

ora::Container ora::Database::getContainer( const std::string& containerName,
                                            const std::type_info&  typeInfo){
  open( true );
  Reflex::Type objType = ClassUtils::lookupDictionary( typeInfo );
  return getContainerFromSession( containerName, objType, *m_impl->m_session );
}

ora::Container ora::Database::getContainer( const std::type_info& typeInfo ){
  open( true );
  Reflex::Type objType = ClassUtils::lookupDictionary( typeInfo );
  std::string contName = nameFromClass( objType );
  return getContainerFromSession( contName, objType, *m_impl->m_session);
}

bool ora::Database::dropContainer( const std::string& name ){
  open();
  if( !m_impl->m_session->containerHandle( name ) ){
    return false;
  }  
  m_impl->m_session->dropContainer( name );
  return true;
}

bool ora::Database::lockContainer( const std::string& name ){
  open();
  Handle<DatabaseContainer> cont = m_impl->m_session->containerHandle( name );
  if( !cont ){
    throwException("Container \""+name+"\" does not exist in the database.",
                   "Database::lockContainer");
  }
  return cont->lock();
}

ora::Container ora::Database::containerHandle( const std::string& name ){
  open();
  Handle<DatabaseContainer> cont = m_impl->m_session->containerHandle( name );
  if( !cont ){
    throwException("Container \""+name+"\" does not exist in the database.",
                   "Database::containerHandle");
  }
  return Container( cont );
}

ora::Container ora::Database::containerHandle( int contId ){
  open();
  Handle<DatabaseContainer> cont = m_impl->m_session->containerHandle( contId );
  if( !cont ){
    std::stringstream messg;
    messg << "Container with id=" << contId << " not found in the database.";
    throwException(messg.str(),
                   "Database::containerHandle");
  }
  return Container( cont );
}

ora::Object ora::Database::fetchItem(const OId& oid){
  Container cont = containerHandle( oid.containerId() );
  return cont.fetchItem( oid.itemId() );
}

ora::OId ora::Database::insertItem(const std::string& containerName,
                                   const Object& dataObject ){
  open( true );  
  Container cont  = getContainerFromSession( containerName, dataObject.type(), *m_impl->m_session );
  int itemId = cont.insertItem( dataObject );
  return OId( cont.id(), itemId );
}

void ora::Database::updateItem(const OId& oid,
                               const Object& dataObject ){
  open();
  Container cont = containerHandle( oid.containerId() );
  cont.updateItem( oid.itemId(), dataObject );
}

void ora::Database::erase(const OId& oid){
  open();
  Container cont = containerHandle( oid.containerId() );
  cont.erase( oid.itemId() );
}

void ora::Database::flush(){
  open();
  const std::map<int,Handle<DatabaseContainer> >& containers = m_impl->m_session->containers();
  for( std::map<int,Handle<DatabaseContainer> >::const_iterator iCont = containers.begin();
       iCont != containers.end(); ++iCont ){
    iCont->second->flush();
  }
}

void ora::Database::setObjectName( const std::string& name, const OId& oid ){
  open( true );
  m_impl->m_session->setObjectName( name, oid.containerId(), oid.itemId() );
}

bool ora::Database::eraseObjectName( const std::string& name ){
  open();
  return m_impl->m_session->eraseObjectName( name );  
}

bool ora::Database::eraseAllNames(){
  open();
  return m_impl->m_session->eraseAllNames();
}

bool ora::Database::getItemId( const std::string& name, ora::OId& destination ){
  open();
  return m_impl->m_session->getItemId( name, destination );  
}

boost::shared_ptr<void> ora::Database::getTypedObjectByName( const std::string& name, const std::type_info& typeInfo ){
  open();
  Reflex::Type objType = ClassUtils::lookupDictionary( typeInfo );
  return m_impl->m_session->fetchTypedObjectByName( name, objType );
}

ora::Object ora::Database::fetchItemByName( const std::string& name ){
  open();
  return  m_impl->m_session->fetchObjectByName( name );
}

bool ora::Database::getNamesForObject( const ora::OId& oid, 
                                       std::vector<std::string>& destination ){
  checkTransaction();
  if( !m_impl->m_session->exists() ){
    throwException("Database does not exists in \""+m_impl->m_session->connectionString()+"\"","Database::getNamesForObject");
  }
  return m_impl->m_session->getNamesForObject( oid.containerId(), oid.itemId(), destination );
}

bool ora::Database::listObjectNames( std::vector<std::string>& destination ){
  checkTransaction();
  if( !m_impl->m_session->exists() ){
    throwException("Database does not exists in \""+m_impl->m_session->connectionString()+"\"","Database::listObjectNames");
  }
  return m_impl->m_session->listObjectNames( destination );
}

ora::DatabaseUtility ora::Database::utility(){
  checkTransaction();
  Handle<DatabaseUtilitySession> utilSession = m_impl->m_session->utility();
  return DatabaseUtility( utilSession );
}

ora::SharedSession& ora::Database::storageAccessSession(){
  return m_impl->m_session->storageAccessSession();
}

