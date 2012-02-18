#include "CondCore/DBCommon/interface/CredentialStore.h"
#include "CondCore/DBCommon/interface/Cipher.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/ORA/interface/SequenceManager.h"
//
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/MessageStream.h"
#include "CoralKernel/Context.h"
#include "RelationalAccess/AuthenticationCredentials.h"
#include "RelationalAccess/IBulkOperation.h"
#include "RelationalAccess/IConnection.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/IRelationalService.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
//
#include "RelationalAccess/AuthenticationCredentials.h"
//
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <boost/filesystem.hpp>

static const std::string serviceName = "CondAuthenticationService";

const std::string coral_bridge::AuthenticationCredentialSet::DEFAULT_ROLE("COND_DEFAULT_ROLE");

coral_bridge::AuthenticationCredentialSet::AuthenticationCredentialSet() :
  m_data(){
}

coral_bridge::AuthenticationCredentialSet::~AuthenticationCredentialSet(){
  for ( std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::iterator iData = m_data.begin();
        iData != m_data.end(); ++iData )
    delete iData->second;
}

void coral_bridge::AuthenticationCredentialSet::registerItem( const std::string& connectionString, 
							      const std::string& itemName,
							      const std::string& itemValue ){
  registerItem( connectionString, DEFAULT_ROLE, itemName, itemValue );
}
			      

void coral_bridge::AuthenticationCredentialSet::registerItem( const std::string& connectionString, 
							      const std::string& role,
							      const std::string& itemName,
							      const std::string& itemValue ){
  std::pair<std::string,std::string> connKey( connectionString, role );
  std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::iterator iData = m_data.find( connKey );
  if( iData == m_data.end() ){
    iData = m_data.insert( std::make_pair( connKey, new coral::AuthenticationCredentials( serviceName ) ) ).first;
  } 
  iData = m_data.insert( std::make_pair( connKey, new coral::AuthenticationCredentials( serviceName ) ) ).first;
  iData->second->registerItem( itemName, itemValue );
}

void
coral_bridge::AuthenticationCredentialSet::registerCredentials( const std::string& connectionString,
								const std::string& userName,
								const std::string& password ){
  registerCredentials( connectionString, DEFAULT_ROLE, userName, password );
}

void
coral_bridge::AuthenticationCredentialSet::registerCredentials( const std::string& connectionString,
								const std::string& role,
								const std::string& userName,
								const std::string& password ){
  std::pair<std::string,std::string> connKey( connectionString, role );
  std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::iterator iData = m_data.find( connKey );
  if( iData != m_data.end() ){
    delete iData->second;
    m_data.erase( connKey );
  } 
  iData = m_data.insert( std::make_pair( connKey, new coral::AuthenticationCredentials( serviceName ) ) ).first;
  iData->second->registerItem( coral::IAuthenticationCredentials::userItem(), userName );
  iData->second->registerItem( coral::IAuthenticationCredentials::passwordItem(), password );
}

void coral_bridge::AuthenticationCredentialSet::import( const AuthenticationCredentialSet& data ){
  for ( std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::const_iterator iData = data.m_data.begin();
        iData != data.m_data.end(); ++iData ){
    registerCredentials( iData->first.first, iData->first.second, iData->second->valueForItem( coral::IAuthenticationCredentials::userItem() ),
			 iData->second->valueForItem( coral::IAuthenticationCredentials::passwordItem() ) );					    
  }
}

const coral::IAuthenticationCredentials*
coral_bridge::AuthenticationCredentialSet::get( const std::string& connectionString ) const
{
  return get( connectionString, DEFAULT_ROLE );
}

const coral::IAuthenticationCredentials*
coral_bridge::AuthenticationCredentialSet::get( const std::string& connectionString, const std::string& role ) const
{
  const coral::IAuthenticationCredentials* ret = 0;
  std::pair<std::string,std::string> connKey( connectionString, role );
  std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::const_iterator iData = m_data.find( connKey );
  if ( iData != m_data.end() ){
    ret = iData->second;
  }
  return ret;
}

const std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >& coral_bridge::AuthenticationCredentialSet::data() const {
  return m_data;
}

namespace cond {
  std::string validPath( const char* envPath ){
    std::string searchPath(envPath);
    boost::filesystem::path fullPath( searchPath );
    if(!boost::filesystem::exists(searchPath) || !boost::filesystem::is_directory( searchPath )){
      throwException( "Authentication Path is invalid.","cond::CredentialStore::validPath" );
    }
    boost::filesystem::path file( DecodingKey::FILE_NAME );
    fullPath /= file;
    return fullPath.string();
  }
}

#define VERIFICATION_KEY TABLE_NAME

static std::string SEQUENCE_TABLE_NAME("COND_CREDENTIAL_SEQUENCE");
static std::string TABLE_NAME("COND_CREDENTIAL");
static std::string ID_COLUMN("ID");
static std::string PRINCIPAL_COLUMN("PRINCIPAL");
static std::string ROLE_COLUMN("ROLE");
static std::string CONNECTION_COLUMN("CRED0");
static std::string USERNAME_COLUMN("CRED1");
static std::string PASSWORD_COLUMN("CRED2");
static std::string VERIFICATION_KEY_COLUMN("CRED3");

const std::string DEFAULT_DATA_SOURCE("Cond_Default_Authentication");

cond::CredentialStore::CoralSession::~CoralSession(){
  session->endUserSession();
  connection->disconnect();
}

// open session on the storage
cond::CredentialStore::CoralSession cond::CredentialStore::openDatabase( bool readMode ){
  coral::MessageStream::setMsgVerbosity( coral::Debug );
  coral::AccessMode accessMode = coral::ReadOnly;
  if(!readMode ) accessMode = coral::Update;
  if(!m_serviceKey){
    std::string msg("");
    msg += "The credential store has not been initialized.";
    throwException( msg,"cond::CredentialStore::openDatabaseForService" );    
  }
  const std::string& storeConnectionString = m_serviceKey->dataSource;
  const std::string& userName = m_serviceKey->userName;
  const std::string& password = m_serviceKey->password;
  coral::IHandle<coral::IRelationalService> relationalService = coral::Context::instance().query<coral::IRelationalService>();
  if ( ! relationalService.isValid() ){
    coral::Context::instance().loadComponent("CORAL/Services/RelationalService");
    relationalService = coral::Context::instance().query<coral::IRelationalService>();
  }
  coral::IRelationalDomain& domain = relationalService->domainForConnection( storeConnectionString );
  std::pair<std::string,std::string> connTokens = domain.decodeUserConnectionString( storeConnectionString );
  CoralSession ret;
  ret.connection.reset( domain.newConnection( connTokens.first ) );
  ret.connection->connect();
  ret.session.reset( ret.connection->newSession( connTokens.second, accessMode) );
  ret.session->startUserSession( userName, password );
  return ret;
}    

cond::CredentialStore::CredentialStore():
  m_serviceKey(0),
  m_key(){
}

cond::CredentialStore::~CredentialStore(){
}

void
cond::CredentialStore::setUpForService( const std::string& serviceName )
{
  if( serviceName.empty() ){
    throwException( "Service name has not been provided.","cond::CredentialStore::setUpConnection" );        
  }
  m_serviceKey = 0;
  const char* authEnv = ::getenv( Auth::COND_AUTH_PATH );
  if(!authEnv){
    authEnv = ::getenv("HOME");
  }
  std::string keyFile = validPath( authEnv );
  m_key.init( keyFile, Auth::COND_KEY ); 
  std::map< std::string, ServiceKey >::const_iterator iK = m_key.serviceKeys().find( serviceName );
  if( iK == m_key.serviceKeys().end() ){
    std::string msg("");
    msg += "Service \""+serviceName+"\" can't be open with the current key.";
    throwException( msg,"cond::CredentialStore::setUpConnection" );    
  }
  m_serviceKey = &iK->second;
}

void 
cond::CredentialStore::setUpForConnectionString( const std::string& connectionString ){
  coral::IHandle<coral::IRelationalService> relationalService = coral::Context::instance().query<coral::IRelationalService>();
  if ( ! relationalService.isValid() ){
    coral::Context::instance().loadComponent("CORAL/Services/RelationalService");
    relationalService = coral::Context::instance().query<coral::IRelationalService>();
  }
  coral::IRelationalDomain& domain = relationalService->domainForConnection( connectionString );
  std::pair<std::string,std::string> connTokens = domain.decodeUserConnectionString( connectionString );
  std::string& serviceName = connTokens.first;
  setUpForService( serviceName );
}


bool
cond::CredentialStore::createSchema()
{
  CoralSession coralDb = openDatabase( false );
  coralDb.session->transaction().start();
  coral::ISchema& schema = coralDb.session->nominalSchema();
  if(schema.existsTable(TABLE_NAME)) {
    coralDb.session->transaction().commit();
    std::stringstream msg;
    msg <<"Cannot create credential database, a table named \"" << TABLE_NAME << "\" already exists.";
    throwException(msg.str(),"CredentialStore::create");
  }
  ora::SequenceManager sequenceMgr( SEQUENCE_TABLE_NAME, schema );
  sequenceMgr.create( TABLE_NAME );
  coral::TableDescription descr;
  descr.setName(TABLE_NAME);
  int columnSize = 2000;
  descr.insertColumn( ID_COLUMN, coral::AttributeSpecification::typeNameForType<int>());
  descr.insertColumn( PRINCIPAL_COLUMN, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr.insertColumn( ROLE_COLUMN, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr.insertColumn( CONNECTION_COLUMN, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr.insertColumn( USERNAME_COLUMN, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr.insertColumn( PASSWORD_COLUMN, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr.insertColumn( VERIFICATION_KEY_COLUMN, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr.setNotNullConstraint( ID_COLUMN );
  descr.setNotNullConstraint( PRINCIPAL_COLUMN );
  descr.setNotNullConstraint( ROLE_COLUMN );
  descr.setNotNullConstraint( CONNECTION_COLUMN );
  descr.setNotNullConstraint( USERNAME_COLUMN );
  descr.setNotNullConstraint( PASSWORD_COLUMN );
  descr.setNotNullConstraint( VERIFICATION_KEY_COLUMN );
  std::vector<std::string> columnsUnique;
  columnsUnique.push_back( PRINCIPAL_COLUMN);
  columnsUnique.push_back( ROLE_COLUMN);
  columnsUnique.push_back( CONNECTION_COLUMN );
  descr.setUniqueConstraint( columnsUnique );
  std::vector<std::string> columnsForIndex;
  columnsForIndex.push_back(ID_COLUMN);
  descr.setPrimaryKey( columnsForIndex );
  schema.createTable( descr );  
  coralDb.session->transaction().commit();
  return true;
}

bool
cond::CredentialStore::drop()
{
  CoralSession coralDb = openDatabase( false );
  coralDb.session->transaction().start();
  coral::ISchema& schema = coralDb.session->nominalSchema();
  if(!schema.existsTable(TABLE_NAME)) {
    coralDb.session->transaction().commit();
    std::stringstream msg;
    msg <<"Cannot drop credential database, a table named \"" << TABLE_NAME << "\" does not exists.";
    throwException(msg.str(),"CredentialStore::drop");
  }
  schema.dropTable( TABLE_NAME );
  coralDb.session->transaction().commit();
  return true;
}


bool
cond::CredentialStore::update( const std::string& principal,
			       const std::string& role,
			       const std::string& connectionString,
			       const std::string& userName,
			       const std::string& password )
{
  CoralSession coralDb = openDatabase( false );
  coralDb.session->transaction().start();
  coral::ISchema& schema = coralDb.session->nominalSchema();
  if(!schema.existsTable(TABLE_NAME)) {
    coralDb.session->transaction().commit();
    throwException( "Cannot find credential database.","CredentialStore::addEntry");
  }
  Cipher cipher( m_serviceKey->key );
  std::string user = cipher.encrypt(userName);
  std::string passwd = cipher.encrypt(password);
  std::string conn = cipher.encrypt(connectionString);

  coral::AttributeList readBuff;
  readBuff.extend<std::string>(CONNECTION_COLUMN);
  std::auto_ptr<coral::IQuery> query(schema.tableHandle(TABLE_NAME).newQuery());
  coral::AttributeList whereData;
  whereData.extend<std::string>(PRINCIPAL_COLUMN);
  whereData.extend<std::string>(ROLE_COLUMN);
  whereData.extend<std::string>(CONNECTION_COLUMN);
  whereData[ PRINCIPAL_COLUMN ].data<std::string>() = principal;
  whereData[ ROLE_COLUMN ].data<std::string>() = role;
  whereData[ CONNECTION_COLUMN ].data<std::string>() = conn;
  std::stringstream whereClause;
  whereClause << PRINCIPAL_COLUMN << " = :"<<PRINCIPAL_COLUMN;
  whereClause << " AND "<<ROLE_COLUMN << " = :"<<ROLE_COLUMN;
  whereClause << " AND "<<CONNECTION_COLUMN << " = :"<<CONNECTION_COLUMN;
  query->defineOutput(readBuff);
  query->setCondition( whereClause.str(), whereData );
  coral::ICursor& cursor = query->execute();
  bool found = false;
  std::string encodedConn("");
  if ( cursor.next() ) {
    found = true;
  }

  if( found ){
    coral::ITableDataEditor& updater = schema.tableHandle(TABLE_NAME).dataEditor();
    coral::AttributeList dataBuffer;
    dataBuffer.extend<std::string>( USERNAME_COLUMN );
    dataBuffer.extend<std::string>( PASSWORD_COLUMN );
    dataBuffer.extend<std::string>( PRINCIPAL_COLUMN );
    dataBuffer.extend<std::string>( CONNECTION_COLUMN );
    dataBuffer[ USERNAME_COLUMN ].data<std::string>() = user;
    dataBuffer[ PASSWORD_COLUMN ].data<std::string>() = passwd;
    dataBuffer[ PRINCIPAL_COLUMN ].data<std::string>() = principal;
    dataBuffer[ CONNECTION_COLUMN ].data<std::string>() = conn;
    std::string setClause = USERNAME_COLUMN+" = :"+USERNAME_COLUMN+", "+
      PASSWORD_COLUMN+" = :"+PASSWORD_COLUMN;
    std::string whereClause = PRINCIPAL_COLUMN+" = :"+PRINCIPAL_COLUMN+
      " AND "+ ROLE_COLUMN+ " = :"+ROLE_COLUMN +
      " AND "+ CONNECTION_COLUMN+" = :"+CONNECTION_COLUMN;
    updater.updateRows( setClause,whereClause, dataBuffer );
    coralDb.session->transaction().commit();
    return true;
  }

  std::string connString = cipher.encrypt( connectionString );
  std::string verifKey = cipher.encrypt(TABLE_NAME);

  ora::SequenceManager sequenceMgr( SEQUENCE_TABLE_NAME,schema );
  int next = sequenceMgr.getNextId( TABLE_NAME, true );

  coral::ITableDataEditor& inserter = schema.tableHandle(TABLE_NAME).dataEditor();
  coral::AttributeList dataBuffer;
  inserter.rowBuffer(dataBuffer);
  dataBuffer[ ID_COLUMN ].data<int>() = next;
  dataBuffer[ PRINCIPAL_COLUMN ].data<std::string>() = principal;
  dataBuffer[ ROLE_COLUMN ].data<std::string>() = role;
  dataBuffer[ CONNECTION_COLUMN ].data<std::string>() = conn;
  dataBuffer[ USERNAME_COLUMN ].data<std::string>() = user;
  dataBuffer[ PASSWORD_COLUMN ].data<std::string>() = passwd;
  dataBuffer[ VERIFICATION_KEY_COLUMN ].data<std::string>() = verifKey;
  inserter.insertRow( dataBuffer );
  coralDb.session->transaction().commit();
  return true;
}

bool 
cond::CredentialStore::remove( const std::string& principal, 
			       const std::string& role,
			       const std::string& connectionString ){
  
  CoralSession coralDb = openDatabase( false );
  coralDb.session->transaction().start();
  coral::ISchema& schema = coralDb.session->nominalSchema();
  if(!schema.existsTable(TABLE_NAME)) {
    coralDb.session->transaction().commit(); 
    throwException("Cannot find credential database.","CredentialStore::removeEntry");
  }

  Cipher cipher( m_serviceKey->key );
  std::string connectionS = cipher.encrypt( connectionString );
  coral::AttributeList dataBuff;
  std::string condition = PRINCIPAL_COLUMN + " = :" + PRINCIPAL_COLUMN + " AND " +
    ROLE_COLUMN+ " = :"+ROLE_COLUMN + " AND "+
    CONNECTION_COLUMN + " = :" + CONNECTION_COLUMN;
  dataBuff.extend<std::string>(PRINCIPAL_COLUMN);
  dataBuff.extend<std::string>(ROLE_COLUMN);
  dataBuff.extend<std::string>(CONNECTION_COLUMN);
  dataBuff[PRINCIPAL_COLUMN].data<std::string>() = principal;
  dataBuff[ROLE_COLUMN].data<std::string>() = role;
  dataBuff[CONNECTION_COLUMN].data<std::string>() = connectionS;
  schema.tableHandle(TABLE_NAME).dataEditor().deleteRows(condition,dataBuff);
  coralDb.session->transaction().commit();
  return true;
}

bool 
cond::CredentialStore::removePrincipal( const std::string& principal ){
  
  CoralSession coralDb = openDatabase( false );
  coralDb.session->transaction().start();
  coral::ISchema& schema = coralDb.session->nominalSchema();
  if(!schema.existsTable(TABLE_NAME)) {
    coralDb.session->transaction().commit();
    throwException("Cannot find credential database.","CredentialStore::removeEntry");
  }

  coral::AttributeList dataBuff;
  std::string condition = PRINCIPAL_COLUMN + " = :" + PRINCIPAL_COLUMN;
  dataBuff.extend<std::string>(PRINCIPAL_COLUMN);
  dataBuff[PRINCIPAL_COLUMN].data<std::string>() = principal;
  schema.tableHandle(TABLE_NAME).dataEditor().deleteRows(condition,dataBuff);
  coralDb.session->transaction().commit();
  return true;
}

bool cond::CredentialStore::exportForPrincipal( const std::string& principal, 
						coral_bridge::AuthenticationCredentialSet& destination ){

  CoralSession coralDb = openDatabase( true );
  coralDb.session->transaction().start( true );
  coral::ISchema& schema = coralDb.session->nominalSchema();
  if(!schema.existsTable(TABLE_NAME)) {
    coralDb.session->transaction().commit();
    throwException("Cannot find credential database.","CredentialStore::removeEntry");
  }
  Cipher cipher( m_serviceKey->key );
  coral::AttributeList readBuff;
  readBuff.extend<std::string>(ROLE_COLUMN);
  readBuff.extend<std::string>(CONNECTION_COLUMN);
  readBuff.extend<std::string>(USERNAME_COLUMN);
  readBuff.extend<std::string>(PASSWORD_COLUMN);
  readBuff.extend<std::string>(VERIFICATION_KEY_COLUMN);
  std::auto_ptr<coral::IQuery> query(schema.tableHandle(TABLE_NAME).newQuery());
  coral::AttributeList whereData;
  whereData.extend<std::string>(PRINCIPAL_COLUMN);
  whereData[PRINCIPAL_COLUMN].data<std::string>() = principal;
  std::string condition = PRINCIPAL_COLUMN + " = :" + PRINCIPAL_COLUMN;
  query->defineOutput(readBuff);
  query->addToOutputList( ROLE_COLUMN );
  query->addToOutputList( CONNECTION_COLUMN );
  query->addToOutputList( USERNAME_COLUMN );
  query->addToOutputList( PASSWORD_COLUMN );
  query->addToOutputList( VERIFICATION_KEY_COLUMN );
  query->setCondition( condition , whereData );
  coral::ICursor& cursor = query->execute();
  bool ret = false;
  while ( cursor.next() ) {
    const coral::AttributeList& row = cursor.currentRow();
    std::string role = row[ROLE_COLUMN].data<std::string>();
    std::string connectionString = cipher.decrypt( row[CONNECTION_COLUMN].data<std::string>() );
    std::string userName = cipher.decrypt( row[USERNAME_COLUMN].data<std::string>() );
    std::string password = cipher.decrypt( row[PASSWORD_COLUMN].data<std::string>() );
    std::string verifKey = cipher.decrypt( row[VERIFICATION_KEY_COLUMN].data<std::string>() );
    if( verifKey == VERIFICATION_KEY ){
      destination.registerCredentials( connectionString, role, userName, password );
      ret = true;
    }
  }
  coralDb.session->transaction().commit();
  return ret;
}

bool cond::CredentialStore::exportAll( coral_bridge::AuthenticationCredentialSet& destination ){
  CoralSession coralDb = openDatabase( true );
  coralDb.session->transaction().start( true );
  coral::ISchema& schema = coralDb.session->nominalSchema();
  if(!schema.existsTable(TABLE_NAME)) {
    coralDb.session->transaction().commit();
    throwException("Cannot find credential database.","CredentialStore::removeEntry");
  }
  Cipher cipher( m_serviceKey->key );
  coral::AttributeList readBuff;
  readBuff.extend<std::string>(PRINCIPAL_COLUMN);
  readBuff.extend<std::string>(ROLE_COLUMN);
  readBuff.extend<std::string>(CONNECTION_COLUMN);
  readBuff.extend<std::string>(USERNAME_COLUMN);
  readBuff.extend<std::string>(PASSWORD_COLUMN);
  readBuff.extend<std::string>(VERIFICATION_KEY_COLUMN);
  std::auto_ptr<coral::IQuery> query(schema.tableHandle(TABLE_NAME).newQuery());
  query->defineOutput(readBuff);
  coral::ICursor& cursor = query->execute();
  bool ret = false;
  const std::string& user = m_key.user();
  coral_bridge::AuthenticationCredentialSet userSet;  
  while ( cursor.next() ) {
    const coral::AttributeList& row = cursor.currentRow();
    std::string principal = row[PRINCIPAL_COLUMN].data<std::string>();
    std::string role = row[ROLE_COLUMN].data<std::string>();
    std::string connectionString = cipher.decrypt( row[CONNECTION_COLUMN].data<std::string>() );
    std::string userName = cipher.decrypt( row[USERNAME_COLUMN].data<std::string>() );
    std::string password = cipher.decrypt( row[PASSWORD_COLUMN].data<std::string>() );
    std::string verifKey = cipher.decrypt( row[VERIFICATION_KEY_COLUMN].data<std::string>() );
    if( !user.empty() && user == principal ){
      if( verifKey == VERIFICATION_KEY ){
	userSet.registerCredentials( connectionString, role, userName, password );
	ret = true;
      }
    } else {
      const std::set<std::string>& groups = m_key.groups();
      std::set<std::string>::const_iterator iG = groups.find( principal );
      if( iG != groups.end() && verifKey == VERIFICATION_KEY ){
	destination.registerCredentials( connectionString, role, userName, password );
	ret = true;
      }
    }
  }
  coralDb.session->transaction().commit();
  destination.import( userSet );
  return ret;
}

bool
cond::CredentialStore::importForPrincipal( const std::string& principal, const coral_bridge::AuthenticationCredentialSet& data ){
  bool imported = false;
  //const std::map< std::string, coral::AuthenticationCredentials* >& defaults = data.defaultCredentials();
  // for the moment only import the roles
  const std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >& creds = data.data();
  for( std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::const_iterator iConn = creds.begin(); iConn != creds.end(); ++iConn ){
    const std::string& connectionString = iConn->first.first;
    const std::string& role = iConn->first.second;
    std::string userName = iConn->second->valueForItem( coral::IAuthenticationCredentials::userItem() );
    std::string password = iConn->second->valueForItem( coral::IAuthenticationCredentials::passwordItem());
    update( principal, role, connectionString, userName, password ); 
    imported = true;
  }
  return imported;
}

