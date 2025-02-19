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
#include "CoralCommon/URIParser.h"
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
#include <sstream>
#include <fstream>
#include <boost/filesystem.hpp>

static const std::string serviceName = "CondAuthenticationService";

coral_bridge::AuthenticationCredentialSet::AuthenticationCredentialSet() :
  m_data(){
}

coral_bridge::AuthenticationCredentialSet::~AuthenticationCredentialSet(){
  reset();
}

void coral_bridge::AuthenticationCredentialSet::reset(){
  for ( std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::iterator iData = m_data.begin();
        iData != m_data.end(); ++iData )
    delete iData->second;
  m_data.clear();
}

void coral_bridge::AuthenticationCredentialSet::registerItem( const std::string& connectionString, 
							      const std::string& itemName,
							      const std::string& itemValue ){
  registerItem( connectionString, cond::Auth::COND_DEFAULT_ROLE, itemName, itemValue );
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
  registerCredentials( connectionString, cond::Auth::COND_DEFAULT_ROLE, userName, password );
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
  return get( connectionString, cond::Auth::COND_DEFAULT_ROLE );
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

static const std::string SEQUENCE_TABLE_NAME("COND_CREDENTIAL_SEQUENCE");

static const std::string COND_AUTHENTICATION_TABLE("COND_AUTHENTICATION");
static const std::string PRINCIPAL_ID_COL("P_ID");
static const std::string PRINCIPAL_NAME_COL("P_NAME");
static const std::string VERIFICATION_COL("CRED0");
static const std::string PRINCIPAL_KEY_COL("CRED1");
static const std::string ADMIN_KEY_COL("CRED2");

static const std::string COND_AUTHORIZATION_TABLE("COND_AUTHORIZATION");
static const std::string AUTH_ID_COL("AUTH_ID");
static const std::string P_ID_COL("P_ID");
static const std::string ROLE_COL("C_ROLE");
static const std::string SCHEMA_COL("C_SCHEMA");
static const std::string AUTH_KEY_COL("CRED3");
static const std::string C_ID_COL("C_ID");

static const std::string COND_CREDENTIAL_TABLE("COND_CREDENTIAL");
static const std::string CONNECTION_ID_COL("CONN_ID");
static const std::string CONNECTION_LABEL_COL("CONN_LABEL");
static const std::string USERNAME_COL("CRED4");
static const std::string PASSWORD_COL("CRED5");
static const std::string VERIFICATION_KEY_COL("CRED6");
static const std::string CONNECTION_KEY_COL("CRED7");

const std::string DEFAULT_DATA_SOURCE("Cond_Default_Authentication");

namespace cond {

  /**
  std::string defaultConnectionString( const std::string& serviceConnectionString, 
				       const std::string& serviceName,
				       const std::string& userName ){
    size_t pos = serviceConnectionString.find( serviceName );
    std::string technologyPrefix = serviceConnectionString.substr(0,pos);
    std::stringstream connStr;
    connStr << technologyPrefix;
    if( !userName.empty() ) connStr <<"/"<< userName;
    return connStr.str();
  }

  **/
  std::string schemaLabel( const std::string& serviceName, 
			   const std::string& userName ){
    std::string ret = userName;
    if( !serviceName.empty() ){
      ret += "@"+serviceName;
    }
    return ret;
  }
  std::string schemaLabelForCredentialStore( const std::string& connectionString ){
    coral::URIParser parser;
    parser.setURI( connectionString );
    std::string serviceName = parser.hostName();
    std::string schemaName = parser.databaseOrSchemaName();
    return schemaLabel( serviceName, schemaName );
  }

  class CSScopedSession {
  public: 
    CSScopedSession( CredentialStore& store ):
      m_store( store ){}
    ~CSScopedSession(){
      m_store.closeSession( false );
    }
    void startSuper( const std::string& connectionString, const std::string& userName, const std::string& password ){
      m_store.startSuperSession( connectionString, userName, password );
    }
    void start( bool readOnly=true ){
      m_store.startSession( readOnly );
    }
    void close(){
      m_store.closeSession();
    }
      
  private:
    CredentialStore& m_store;
  };

  struct PrincipalData {
    int id;
    std::string verifKey;
    std::string principalKey;
    std::string adminKey;
    PrincipalData():
      id(-1),
      verifKey(""),
      principalKey(""),
      adminKey(""){}
  };

  bool selectPrincipal( coral::ISchema& schema, const std::string& principal, PrincipalData& destination ){
    std::auto_ptr<coral::IQuery> query(schema.tableHandle(COND_AUTHENTICATION_TABLE).newQuery());
    coral::AttributeList readBuff;
    readBuff.extend<int>(PRINCIPAL_ID_COL);
    readBuff.extend<std::string>(VERIFICATION_COL);
    readBuff.extend<std::string>(PRINCIPAL_KEY_COL);
    readBuff.extend<std::string>(ADMIN_KEY_COL);
    coral::AttributeList whereData;
    whereData.extend<std::string>(PRINCIPAL_NAME_COL);
    whereData[ PRINCIPAL_NAME_COL ].data<std::string>() = principal;
    std::string whereClause = PRINCIPAL_NAME_COL + " = :" + PRINCIPAL_NAME_COL;
    query->defineOutput(readBuff);
    query->addToOutputList( PRINCIPAL_ID_COL );
    query->addToOutputList( VERIFICATION_COL );
    query->addToOutputList( PRINCIPAL_KEY_COL );
    query->addToOutputList( ADMIN_KEY_COL );
    query->setCondition( whereClause, whereData );
    coral::ICursor& cursor = query->execute();
    bool found = false;
    if ( cursor.next() ) {
      found = true;
      const coral::AttributeList& row = cursor.currentRow();
      destination.id = row[ PRINCIPAL_ID_COL ].data<int>();
      destination.verifKey = row[ VERIFICATION_COL ].data<std::string>();
      destination.principalKey = row[ PRINCIPAL_KEY_COL ].data<std::string>();
      destination.adminKey = row[ ADMIN_KEY_COL ].data<std::string>();
    }
    return found;
  }

  struct CredentialData {
    int id;
    std::string userName;
    std::string password;
    std::string connectionKey;
    std::string verificationKey;
    CredentialData():
      id(-1),
      userName(""),
      password(""),
      connectionKey(""){
    }
  };

  bool selectConnection( coral::ISchema& schema, const std::string& connectionLabel, CredentialData& destination ){
    
    std::auto_ptr<coral::IQuery> query(schema.tableHandle(COND_CREDENTIAL_TABLE).newQuery());
    coral::AttributeList readBuff;
    readBuff.extend<int>( CONNECTION_ID_COL );
    readBuff.extend<std::string>( USERNAME_COL );
    readBuff.extend<std::string>( PASSWORD_COL );
    readBuff.extend<std::string>( VERIFICATION_KEY_COL );
    readBuff.extend<std::string>( CONNECTION_KEY_COL );
    coral::AttributeList whereData;
    whereData.extend<std::string>( CONNECTION_LABEL_COL );
    whereData[ CONNECTION_LABEL_COL ].data<std::string>() = connectionLabel;
    std::string whereClause = CONNECTION_LABEL_COL + " = :" + CONNECTION_LABEL_COL;
    query->defineOutput(readBuff);
    query->addToOutputList( CONNECTION_ID_COL );
    query->addToOutputList( USERNAME_COL );
    query->addToOutputList( PASSWORD_COL );
    query->addToOutputList( VERIFICATION_KEY_COL );
    query->addToOutputList( CONNECTION_KEY_COL );
    query->setCondition( whereClause, whereData );
    coral::ICursor& cursor = query->execute();
    bool found = false;
    if ( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      destination.id = row[ CONNECTION_ID_COL].data<int>();
      destination.userName = row[ USERNAME_COL].data<std::string>();
      destination.password = row[ PASSWORD_COL].data<std::string>();
      destination.verificationKey = row[ VERIFICATION_KEY_COL].data<std::string>();
      destination.connectionKey = row[ CONNECTION_KEY_COL].data<std::string>();
      found = true;
    }
    return found;
  }

  struct AuthorizationData {
    int id;
    int connectionId;
    std::string key;
    AuthorizationData():
      id(-1),
      connectionId(-1),
      key(""){}
  };

  bool selectAuthorization( coral::ISchema& schema, int principalId, const std::string& role, const std::string& connectionString, AuthorizationData& destination ){
    std::auto_ptr<coral::IQuery> query(schema.tableHandle(COND_AUTHORIZATION_TABLE).newQuery());
    coral::AttributeList readBuff;
    readBuff.extend<int>(AUTH_ID_COL);
    readBuff.extend<int>(C_ID_COL);
    readBuff.extend<std::string>(AUTH_KEY_COL);
    coral::AttributeList whereData;
    whereData.extend<int>(P_ID_COL);
    whereData.extend<std::string>(ROLE_COL);
    whereData.extend<std::string>(SCHEMA_COL);
    whereData[ P_ID_COL ].data<int>() = principalId;
    whereData[ ROLE_COL ].data<std::string>() = role;
    whereData[ SCHEMA_COL ].data<std::string>() = connectionString;
    std::stringstream whereClause;
    whereClause << P_ID_COL << " = :"<<P_ID_COL;
    whereClause << " AND " << ROLE_COL << " = :"<<ROLE_COL;
    whereClause << " AND " << SCHEMA_COL << " = :"<<SCHEMA_COL;
    query->defineOutput(readBuff);
    query->addToOutputList( AUTH_ID_COL );
    query->addToOutputList( C_ID_COL );
    query->addToOutputList( AUTH_KEY_COL );
    query->setCondition( whereClause.str(), whereData );
    coral::ICursor& cursor = query->execute();
    bool found = false;
    if ( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      destination.id = row[AUTH_ID_COL].data<int>();  
      destination.connectionId = row[C_ID_COL].data<int>();
      destination.key = row[AUTH_KEY_COL].data<std::string>();
      found = true;
    }
    return found;
  }

}

void cond::CredentialStore::closeSession( bool commit ){

  if( m_session.get() ){
    if(m_session->transaction().isActive()){
      if( commit ){
	m_session->transaction().commit();
      } else {
	m_session->transaction().rollback();
      }
    }
    m_session->endUserSession();
  }
  m_session.reset();
  if( m_connection.get() ){
    m_connection->disconnect();
  }
  m_connection.reset();
}

int cond::CredentialStore::addUser( const std::string& principalName, const std::string& authenticationKey, const std::string& principalKey, const std::string& adminKey ){

  coral::ISchema& schema = m_session->nominalSchema();
  coral::ITableDataEditor& editor0 = schema.tableHandle(COND_AUTHENTICATION_TABLE).dataEditor();

  ora::SequenceManager sequenceMgr( SEQUENCE_TABLE_NAME,schema );
  int principalId = sequenceMgr.getNextId( COND_AUTHENTICATION_TABLE, true );
    
  Cipher cipher0( authenticationKey );
  Cipher cipher1( adminKey );

  coral::AttributeList authData;
  editor0.rowBuffer(authData);
  authData[ PRINCIPAL_ID_COL ].data<int>() = principalId;
  authData[ PRINCIPAL_NAME_COL ].data<std::string>() = principalName;
  authData[ VERIFICATION_COL ].data<std::string>() = cipher0.b64encrypt( principalName );
  authData[ PRINCIPAL_KEY_COL ].data<std::string>() = cipher0.b64encrypt( principalKey );
  authData[ ADMIN_KEY_COL ].data<std::string>() = cipher1.b64encrypt( principalKey );
  editor0.insertRow( authData );
  return principalId;
}

std::pair<std::string,std::string> cond::CredentialStore::openConnection( const std::string& connectionString ){
  coral::IHandle<coral::IRelationalService> relationalService = coral::Context::instance().query<coral::IRelationalService>();
  if ( ! relationalService.isValid() ){
    coral::Context::instance().loadComponent("CORAL/Services/RelationalService");
    relationalService = coral::Context::instance().query<coral::IRelationalService>();
  }
  coral::IRelationalDomain& domain = relationalService->domainForConnection( connectionString );
  std::pair<std::string,std::string> connTokens = domain.decodeUserConnectionString( connectionString );
  m_connection.reset( domain.newConnection( connTokens.first ) );
  m_connection->connect();
  return connTokens;
}

void cond::CredentialStore::openSession( const std::string& schemaName, const std::string& userName, const std::string& password, bool readMode ){
  coral::AccessMode accessMode = coral::ReadOnly;
  if( !readMode ) accessMode = coral::Update;  
  m_session.reset( m_connection->newSession( schemaName, accessMode) );
  m_session->startUserSession( userName, password );
  // open read-only transaction
  m_session->transaction().start( readMode );
}

void cond::CredentialStore::startSuperSession( const std::string& connectionString, const std::string& userName, const std::string& password ){
  std::pair<std::string,std::string> connTokens = openConnection( connectionString );
  openSession( connTokens.second, userName, password, false );
}

// open session on the storage
void cond::CredentialStore::startSession( bool readMode ){
  if(!m_serviceData){
    throwException( "The credential store has not been initialized.","cond::CredentialStore::openConnection" );    
  }
  const std::string& storeConnectionString = m_serviceData->connectionString;

  std::pair<std::string,std::string> connTokens = openConnection( storeConnectionString );

  const std::string& userName = m_serviceData->userName;
  const std::string& password = m_serviceData->password;
 
  openSession( connTokens.second, userName, password, true );

  coral::ISchema& schema = m_session->nominalSchema();
  if(!schema.existsTable(COND_AUTHENTICATION_TABLE) ||
     !schema.existsTable(COND_AUTHORIZATION_TABLE) ||
     !schema.existsTable(COND_CREDENTIAL_TABLE) ){
    throwException("Credential database does not exists in \""+storeConnectionString+"\"","CredentialStore::startSession");
  }

  const std::string& principalName = m_key.principalName();
  // now authenticate...
  PrincipalData princData;
  if( !selectPrincipal( m_session->nominalSchema(), principalName, princData ) ){
    throwException( "Invalid credentials provided.(0)",
		    "CredentialStore::openSession");
  }
  Cipher cipher0( m_key.principalKey() );
  std::string verifStr = cipher0.b64decrypt( princData.verifKey );
  if( verifStr != principalName ){
    throwException( "Invalid credentials provided (1)",
		    "CredentialStore::openSession");
  }
  // ok, authenticated!
  m_principalId = princData.id;
  m_principalKey = cipher0.b64decrypt( princData.principalKey );
  
  if(!readMode ) {

    Cipher cipher0( m_principalKey );
    std::string adminKey = cipher0.b64decrypt( princData.adminKey );
    if( adminKey != m_principalKey ){
      // not admin user!
      throwException( "Provided credentials does not allow admin operation.",
		      "CredentialStore::openSession");
    }
    
    // first find the credentials for WRITING in the security tables
    std::auto_ptr<coral::IQuery> query(schema.newQuery());
    query->addToTableList(COND_AUTHORIZATION_TABLE, "AUTHO");
    query->addToTableList(COND_CREDENTIAL_TABLE, "CREDS");
    coral::AttributeList readBuff;
    readBuff.extend<std::string>("CREDS."+CONNECTION_LABEL_COL);
    readBuff.extend<std::string>("CREDS."+CONNECTION_KEY_COL);
    readBuff.extend<std::string>("CREDS."+USERNAME_COL);
    readBuff.extend<std::string>("CREDS."+PASSWORD_COL);
    readBuff.extend<std::string>("CREDS."+VERIFICATION_KEY_COL);
    coral::AttributeList whereData;
    whereData.extend<int>(P_ID_COL);
    whereData.extend<std::string>(ROLE_COL);
    whereData.extend<std::string>(SCHEMA_COL);
    whereData[ P_ID_COL ].data<int>() = m_principalId;
    whereData[ ROLE_COL ].data<std::string>() = Auth::COND_ADMIN_ROLE;
    whereData[ SCHEMA_COL ].data<std::string>() = storeConnectionString;
    std::stringstream whereClause;
    whereClause << "AUTHO."<< C_ID_COL << " = CREDS."<<CONNECTION_ID_COL;
    whereClause << " AND AUTHO."<< P_ID_COL << " = :"<<P_ID_COL;
    whereClause << " AND AUTHO."<< ROLE_COL << " = :"<<ROLE_COL;
    whereClause << " AND AUTHO."<< SCHEMA_COL << " = :"<<SCHEMA_COL;
    query->defineOutput(readBuff);
    query->addToOutputList( "CREDS."+CONNECTION_LABEL_COL );
    query->addToOutputList( "CREDS."+CONNECTION_KEY_COL );
    query->addToOutputList( "CREDS."+USERNAME_COL );
    query->addToOutputList( "CREDS."+PASSWORD_COL );
    query->addToOutputList( "CREDS."+VERIFICATION_KEY_COL );
    query->setCondition( whereClause.str(), whereData );
    coral::ICursor& cursor = query->execute();
    bool found = false;
    std::string writeUserName("");
    std::string writePassword("");
    if ( cursor.next() ) {
      const coral::AttributeList& row = cursor.currentRow();
      const std::string& connLabel = row[ "CREDS."+CONNECTION_LABEL_COL ].data<std::string>();
      const std::string& encryptedConnectionKey = row[ "CREDS."+CONNECTION_KEY_COL ].data<std::string>();
      std::string connectionKey = cipher0.b64decrypt( encryptedConnectionKey );
      Cipher cipher1( connectionKey );
      const std::string& encryptedUserName = row[ "CREDS."+USERNAME_COL ].data<std::string>();
      const std::string& encryptedPassword = row[ "CREDS."+PASSWORD_COL ].data<std::string>();
      const std::string& verificationKey = row[ "CREDS."+VERIFICATION_KEY_COL ].data<std::string>();
      if( cipher1.b64decrypt( verificationKey ) != connLabel  ){
	throwException( "Could not decrypt credentials.Provided key is invalid.",
			"CredentialStore::startSession");
      }
      writeUserName = cipher1.b64decrypt( encryptedUserName );
      writePassword = cipher1.b64decrypt( encryptedPassword );
      found = true;
    }
    if( ! found ){
      throwException( "Provided credentials are invalid for write access.",
		      "CredentialStore::openSession");
    }
    m_session->transaction().commit();
    m_session->endUserSession();
    openSession( connTokens.second, writeUserName, writePassword, false );

  }
}   

bool cond::CredentialStore::setPermission( int principalId, const std::string& principalKey, const std::string& role, const std::string& connectionString, int connectionId, const std::string& connectionKey ){
  coral::ISchema& schema = m_session->nominalSchema();
  Cipher cipher( principalKey );
  std::string encryptedConnectionKey = cipher.b64encrypt( connectionKey );

  AuthorizationData authData;
  bool found = selectAuthorization( schema, principalId, role, connectionString, authData );

  coral::ITableDataEditor& editor = schema.tableHandle(COND_AUTHORIZATION_TABLE).dataEditor();
  if( found ) {
    coral::AttributeList updateData;
    updateData.extend<int>( AUTH_ID_COL );
    updateData.extend<int>( C_ID_COL );
    updateData.extend<std::string>( AUTH_KEY_COL );
    updateData[ AUTH_ID_COL ].data<int>() = authData.id;
    updateData[ C_ID_COL ].data<int>() = connectionId;
    updateData[ AUTH_KEY_COL ].data<std::string>() = encryptedConnectionKey;
    std::string setCl = C_ID_COL+" = :"+C_ID_COL + ", "+AUTH_KEY_COL+" = :"+AUTH_KEY_COL;
    std::string whereCl = AUTH_ID_COL+" = :"+AUTH_ID_COL;
    editor.updateRows( setCl,whereCl, updateData );
  } else {

    ora::SequenceManager sequenceMgr( SEQUENCE_TABLE_NAME,schema );
    int next = sequenceMgr.getNextId( COND_AUTHORIZATION_TABLE, true );
    
    coral::AttributeList insertData;
    insertData.extend<int>( AUTH_ID_COL );
    insertData.extend<int>( P_ID_COL );
    insertData.extend<std::string>( ROLE_COL );
    insertData.extend<std::string>( SCHEMA_COL );
    insertData.extend<std::string>( AUTH_KEY_COL );
    insertData.extend<int>( C_ID_COL );
    insertData[ AUTH_ID_COL ].data<int>() = next;
    insertData[ P_ID_COL ].data<int>() = principalId;
    insertData[ ROLE_COL ].data<std::string>() = role;
    insertData[ SCHEMA_COL ].data<std::string>() = connectionString;
    insertData[ AUTH_KEY_COL ].data<std::string>() = encryptedConnectionKey;
    insertData[ C_ID_COL ].data<int>() = connectionId;
    editor.insertRow( insertData );
  }
  return true;    
}

std::pair<int,std::string> cond::CredentialStore::updateConnection( const std::string& connectionLabel, 
								    const std::string& userName, 
								    const std::string& password,
								    bool forceUpdate ){
  coral::ISchema& schema = m_session->nominalSchema();
  CredentialData credsData;
  bool found = selectConnection( schema, connectionLabel, credsData );
  int connId = credsData.id;
  
  Cipher adminCipher( m_principalKey );
  std::string connectionKey("");
  coral::ITableDataEditor& editor = schema.tableHandle(COND_CREDENTIAL_TABLE).dataEditor();
  if( found ){
    
    connectionKey = adminCipher.b64decrypt( credsData.connectionKey );
    Cipher cipher( connectionKey );
    std::string verificationKey = cipher.b64decrypt( credsData.verificationKey );
    if( verificationKey != connectionLabel ){
      throwException("Decoding of connection key failed.","CredentialStore::updateConnection");
    }
    if( forceUpdate ){
      std::string encryptedUserName = cipher.b64encrypt( userName );
      std::string encryptedPassword = cipher.b64encrypt( password );
     
      coral::AttributeList updateData;
      updateData.extend<int>( CONNECTION_ID_COL );
      updateData.extend<std::string>( USERNAME_COL );
      updateData.extend<std::string>( PASSWORD_COL );
      updateData[ CONNECTION_ID_COL ].data<int>() = connId;
      updateData[ USERNAME_COL ].data<std::string>() = encryptedUserName;
      updateData[ PASSWORD_COL ].data<std::string>() = encryptedPassword;
      std::stringstream setCl;
      setCl << USERNAME_COL << " = :" << USERNAME_COL;
      setCl <<", " << PASSWORD_COL << " = :" << PASSWORD_COL;
      std::string whereCl = CONNECTION_ID_COL+" = :"+CONNECTION_ID_COL;
      editor.updateRows( setCl.str(),whereCl, updateData );
    }
  }
  
  if(!found){
    
    KeyGenerator gen;
    connectionKey = gen.make( Auth::COND_DB_KEY_SIZE );
    Cipher cipher( connectionKey );
    std::string encryptedUserName = cipher.b64encrypt( userName );
    std::string encryptedPassword = cipher.b64encrypt( password );
    std::string encryptedLabel = cipher.b64encrypt( connectionLabel );
    
    ora::SequenceManager sequenceMgr( SEQUENCE_TABLE_NAME,schema );
    connId = sequenceMgr.getNextId( COND_CREDENTIAL_TABLE, true );
    
    coral::AttributeList insertData;
    insertData.extend<int>( CONNECTION_ID_COL );
    insertData.extend<std::string>( CONNECTION_LABEL_COL );
    insertData.extend<std::string>( USERNAME_COL );
    insertData.extend<std::string>( PASSWORD_COL );
    insertData.extend<std::string>( VERIFICATION_KEY_COL );
    insertData.extend<std::string>( CONNECTION_KEY_COL );
    insertData[ CONNECTION_ID_COL ].data<int>() = connId;
    insertData[ CONNECTION_LABEL_COL ].data<std::string>() = connectionLabel;
    insertData[ USERNAME_COL ].data<std::string>() = encryptedUserName;
    insertData[ PASSWORD_COL ].data<std::string>() = encryptedPassword;
    insertData[ VERIFICATION_KEY_COL ].data<std::string>() = encryptedLabel;
    insertData[ CONNECTION_KEY_COL ].data<std::string>() = adminCipher.b64encrypt( connectionKey ) ;;
    editor.insertRow( insertData );
    
    /***
    // then set the admin permission on the new connection
    ora::SequenceManager sequenceMgr2( SEQUENCE_TABLE_NAME,schema );
    int authId = sequenceMgr2.getNextId( COND_AUTHORIZATION_TABLE, true );
    
    coral::ITableDataEditor& authEditor = schema.tableHandle(COND_AUTHORIZATION_TABLE).dataEditor();
    coral::AttributeList authData;
    authData.extend<int>( AUTH_ID_COL );
    authData.extend<int>( P_ID_COL );
    authData.extend<std::string>( ROLE_COL );
    authData.extend<std::string>( SCHEMA_COL );
    authData.extend<std::string>( AUTH_KEY_COL );
    authData.extend<int>( C_ID_COL );
    authData[ AUTH_ID_COL ].data<int>() = authId;
    authData[ P_ID_COL ].data<int>() = m_principalId;
    authData[ ROLE_COL ].data<std::string>() = Auth::COND_ADMIN_ROLE;
    authData[ SCHEMA_COL ].data<std::string>() = defaultConnectionString( m_serviceData->connectionString, m_serviceName, userName );
    authData[ AUTH_KEY_COL ].data<std::string>() = adminCipher.b64encrypt( connectionKey ) ;
    authData[ C_ID_COL ].data<int>() = connId;
    authEditor.insertRow( authData );
    **/
  }
  return std::make_pair( connId, connectionKey );
}

cond::CredentialStore::CredentialStore():
  m_connection(),
  m_session(),
  m_principalId(-1),
  m_principalKey(""),
  m_serviceName(""),
  m_serviceData(0),
  m_key(){
}

cond::CredentialStore::~CredentialStore(){
}

std::string
cond::CredentialStore::setUpForService( const std::string& serviceName, 
					const std::string& authPath ){
  if( serviceName.empty() ){
    throwException( "Service name has not been provided.","cond::CredentialStore::setUpConnection" );        
  }
  m_serviceName.clear();
  m_serviceData = 0;

  if( authPath.empty() ){
    throwException( "The authentication Path has not been provided.","cond::CredentialStore::setUpForService" );
  }
  boost::filesystem::path fullPath( authPath );
  if(!boost::filesystem::exists(authPath) || !boost::filesystem::is_directory( authPath )){
    throwException( "Authentication Path is invalid.","cond::CredentialStore::setUpForService" );
  }
  boost::filesystem::path file( DecodingKey::FILE_PATH );
  fullPath /= file;

  m_key.init( fullPath.string(), Auth::COND_KEY ); 
  
  std::map< std::string, ServiceCredentials >::const_iterator iK = m_key.services().find( serviceName );
  if( iK == m_key.services().end() ){
    std::string msg("");
    msg += "Service \""+serviceName+"\" can't be open with the current key.";
    throwException( msg,"cond::CredentialStore::setUpConnection" );    
  }
  m_serviceName = serviceName;
  m_serviceData = &iK->second;
  return m_serviceData->connectionString;
}

std::string
cond::CredentialStore::setUpForConnectionString( const std::string& connectionString,
						 const std::string& authPath ){
  coral::IHandle<coral::IRelationalService> relationalService = coral::Context::instance().query<coral::IRelationalService>();
  if ( ! relationalService.isValid() ){
    coral::Context::instance().loadComponent("CORAL/Services/RelationalService");
    relationalService = coral::Context::instance().query<coral::IRelationalService>();
  }
  coral::IRelationalDomain& domain = relationalService->domainForConnection( connectionString );
  std::pair<std::string,std::string> connTokens = domain.decodeUserConnectionString( connectionString );
  std::string& serviceName = connTokens.first;
  return setUpForService( serviceName, authPath );
}


bool
cond::CredentialStore::createSchema( const std::string& connectionString, const std::string& userName, const std::string& password ) {
  CSScopedSession session( *this );
  session.startSuper( connectionString, userName, password );

  coral::ISchema& schema = m_session->nominalSchema();
  if(schema.existsTable(COND_AUTHENTICATION_TABLE)) {
    throwException("Credential database, already exists.","CredentialStore::create");
  }
  ora::SequenceManager sequenceMgr( SEQUENCE_TABLE_NAME, schema );
  int columnSize = 2000;

  // authentication table
  sequenceMgr.create( COND_AUTHENTICATION_TABLE );
  coral::TableDescription descr0;
  descr0.setName( COND_AUTHENTICATION_TABLE );
  descr0.insertColumn( PRINCIPAL_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr0.insertColumn( PRINCIPAL_NAME_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr0.insertColumn( VERIFICATION_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr0.insertColumn( PRINCIPAL_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr0.insertColumn( ADMIN_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr0.setNotNullConstraint( PRINCIPAL_ID_COL );
  descr0.setNotNullConstraint( PRINCIPAL_NAME_COL );
  descr0.setNotNullConstraint( VERIFICATION_COL );
  descr0.setNotNullConstraint( PRINCIPAL_KEY_COL );
  descr0.setNotNullConstraint( ADMIN_KEY_COL );
  std::vector<std::string> columnsUnique;
  columnsUnique.push_back( PRINCIPAL_NAME_COL);
  descr0.setUniqueConstraint( columnsUnique );
  std::vector<std::string> columnsForIndex;
  columnsForIndex.push_back(PRINCIPAL_ID_COL);
  descr0.setPrimaryKey( columnsForIndex );
  schema.createTable( descr0 );

  // authorization table
  sequenceMgr.create( COND_AUTHORIZATION_TABLE );
  coral::TableDescription descr1;
  descr1.setName( COND_AUTHORIZATION_TABLE );
  descr1.insertColumn( AUTH_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr1.insertColumn( P_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr1.insertColumn( ROLE_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr1.insertColumn( SCHEMA_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr1.insertColumn( AUTH_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr1.insertColumn( C_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr1.setNotNullConstraint( AUTH_ID_COL );
  descr1.setNotNullConstraint( P_ID_COL );
  descr1.setNotNullConstraint( ROLE_COL );
  descr1.setNotNullConstraint( SCHEMA_COL );
  descr1.setNotNullConstraint( AUTH_KEY_COL );
  descr1.setNotNullConstraint( C_ID_COL );
  columnsUnique.clear();
  columnsUnique.push_back( P_ID_COL);
  columnsUnique.push_back( ROLE_COL);
  columnsUnique.push_back( SCHEMA_COL);
  descr1.setUniqueConstraint( columnsUnique );
  columnsForIndex.clear();
  columnsForIndex.push_back(AUTH_ID_COL);
  descr1.setPrimaryKey( columnsForIndex );
  schema.createTable( descr1 );

  // credential table
  sequenceMgr.create( COND_CREDENTIAL_TABLE );
  coral::TableDescription descr2;
  descr2.setName( COND_CREDENTIAL_TABLE );
  descr2.insertColumn( CONNECTION_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr2.insertColumn( CONNECTION_LABEL_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr2.insertColumn( USERNAME_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr2.insertColumn( PASSWORD_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr2.insertColumn( VERIFICATION_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr2.insertColumn( CONNECTION_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(),columnSize,false);
  descr2.setNotNullConstraint( CONNECTION_ID_COL );
  descr2.setNotNullConstraint( CONNECTION_LABEL_COL );
  descr2.setNotNullConstraint( USERNAME_COL );
  descr2.setNotNullConstraint( PASSWORD_COL );
  descr2.setNotNullConstraint( VERIFICATION_KEY_COL );
  descr2.setNotNullConstraint( CONNECTION_KEY_COL );
  columnsUnique.clear();
  columnsUnique.push_back( CONNECTION_LABEL_COL);
  descr2.setUniqueConstraint( columnsUnique );
  columnsForIndex.clear();
  columnsForIndex.push_back(CONNECTION_ID_COL);
  descr2.setPrimaryKey( columnsForIndex );
  schema.createTable( descr2 );

  session.close();
  return true;
}

bool
cond::CredentialStore::drop( const std::string& connectionString, const std::string& userName, const std::string& password ) {
  CSScopedSession session( *this );
  session.startSuper( connectionString, userName, password );

  coral::ISchema& schema = m_session->nominalSchema();
  schema.dropIfExistsTable( COND_AUTHORIZATION_TABLE );
  schema.dropIfExistsTable( COND_CREDENTIAL_TABLE );
  schema.dropIfExistsTable( COND_AUTHENTICATION_TABLE );
  schema.dropIfExistsTable(SEQUENCE_TABLE_NAME);
  session.close();
  return true;
}

bool cond::CredentialStore::installAdmin( const std::string& userName, const std::string& password ){
  if(!m_serviceData){
    throwException( "The credential store has not been initialized.","cond::CredentialStore::installAdmin" );    
  }
  const std::string& connectionString = m_serviceData->connectionString;
  const std::string& principalName = m_key.principalName();

  CSScopedSession session( *this );
  session.startSuper( connectionString, userName, password  );

  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal( schema, principalName, princData );

  if( found ){
    std::string msg("Principal \"");
    msg += principalName + "\" has been installed already.";
    throwException(msg,"CredentialStore::installAdmin");
  }

  KeyGenerator gen;
  m_principalKey = gen.make( Auth::COND_DB_KEY_SIZE );

  coral::ITableDataEditor& editor0 = schema.tableHandle(COND_AUTHENTICATION_TABLE).dataEditor();

  ora::SequenceManager sequenceMgr( SEQUENCE_TABLE_NAME,schema );
  int principalId = sequenceMgr.getNextId( COND_AUTHENTICATION_TABLE, true );
    
  Cipher cipher0( m_key.principalKey() );
  Cipher cipher1( m_principalKey );

  coral::AttributeList authData;
  editor0.rowBuffer(authData);
  authData[ PRINCIPAL_ID_COL ].data<int>() = principalId;
  authData[ PRINCIPAL_NAME_COL ].data<std::string>() = principalName;
  authData[ VERIFICATION_COL ].data<std::string>() = cipher0.b64encrypt( principalName );
  authData[ PRINCIPAL_KEY_COL ].data<std::string>() = cipher0.b64encrypt( m_principalKey );
  authData[ ADMIN_KEY_COL ].data<std::string>() = cipher1.b64encrypt( m_principalKey );
  editor0.insertRow( authData );

  std::string connLabel = schemaLabelForCredentialStore( connectionString );
  DecodingKey tmpKey;
  std::string connectionKey = gen.make( Auth::COND_DB_KEY_SIZE );
  std::string encryptedConnectionKey = cipher1.b64encrypt( connectionKey );    

  Cipher cipher2( connectionKey );
  std::string encryptedUserName = cipher2.b64encrypt( userName );
  std::string encryptedPassword = cipher2.b64encrypt( password );
  std::string encryptedLabel = cipher2.b64encrypt( connLabel );    
  
  int connId = sequenceMgr.getNextId( COND_CREDENTIAL_TABLE, true );
    
  coral::ITableDataEditor& editor1 = schema.tableHandle(COND_CREDENTIAL_TABLE).dataEditor();
  coral::AttributeList connectionData;
  editor1.rowBuffer(connectionData);
  connectionData[ CONNECTION_ID_COL ].data<int>() = connId;
  connectionData[ CONNECTION_LABEL_COL ].data<std::string>() = connLabel;
  connectionData[ USERNAME_COL ].data<std::string>() = encryptedUserName;
  connectionData[ PASSWORD_COL ].data<std::string>() = encryptedPassword;
  connectionData[ VERIFICATION_KEY_COL ].data<std::string>() = encryptedLabel;
  connectionData[ CONNECTION_KEY_COL ].data<std::string>() = encryptedConnectionKey;
  editor1.insertRow( connectionData );

  int authId = sequenceMgr.getNextId( COND_AUTHORIZATION_TABLE, true );
    
  coral::ITableDataEditor& editor2 = schema.tableHandle(COND_AUTHORIZATION_TABLE).dataEditor();
  coral::AttributeList permissionData;
  editor2.rowBuffer(permissionData);
  permissionData[ AUTH_ID_COL ].data<int>() = authId;
  permissionData[ P_ID_COL ].data<int>() = principalId;
  permissionData[ ROLE_COL ].data<std::string>() = Auth::COND_ADMIN_ROLE;
  permissionData[ SCHEMA_COL ].data<std::string>() = connectionString;
  permissionData[ AUTH_KEY_COL ].data<std::string>() = encryptedConnectionKey;
  permissionData[ C_ID_COL ].data<int>() = connId;
  editor2.insertRow( permissionData );

  session.close();
  return true;
}

bool cond::CredentialStore::updatePrincipal( const std::string& principalName, 
					     const std::string& authenticationKey,
					     bool setAdmin ){
  CSScopedSession session( *this );
  session.start( false  );

  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal( schema, principalName, princData );

  Cipher adminCipher( m_principalKey );
  Cipher cipher( authenticationKey );
  std::string verifStr = cipher.b64encrypt( principalName );
  std::string principalKey("");
  if( setAdmin ) principalKey = m_principalKey;
  int principalId = princData.id;

  coral::ITableDataEditor& editor = schema.tableHandle(COND_AUTHENTICATION_TABLE).dataEditor();
  if( found ){
    if( principalKey.empty() ) principalKey = adminCipher.b64decrypt( princData.adminKey );
    coral::AttributeList updateData;
    updateData.extend<int>( PRINCIPAL_ID_COL );
    updateData.extend<std::string>( VERIFICATION_COL );
    updateData.extend<std::string>( PRINCIPAL_KEY_COL );
    updateData.extend<std::string>( ADMIN_KEY_COL );
    updateData[ PRINCIPAL_ID_COL ].data<int>() = principalId;
    updateData[ VERIFICATION_COL ].data<std::string>() = verifStr;
    updateData[ PRINCIPAL_KEY_COL ].data<std::string>() = cipher.b64encrypt( principalKey );
    updateData[ ADMIN_KEY_COL ].data<std::string>() = adminCipher.b64encrypt( principalKey );
    std::stringstream setClause;
    setClause << VERIFICATION_COL <<" = :" <<VERIFICATION_COL <<", ";
    setClause << PRINCIPAL_KEY_COL << " = :" << PRINCIPAL_KEY_COL <<", ";
    setClause << ADMIN_KEY_COL << " = :" << ADMIN_KEY_COL;
    std::string whereClause = PRINCIPAL_ID_COL+" = :"+PRINCIPAL_ID_COL;
    editor.updateRows( setClause.str(),whereClause, updateData );
  } else {
    if( principalKey.empty() ) {
      KeyGenerator gen;
      principalKey = gen.make( Auth::COND_DB_KEY_SIZE );
    }

    coral::ITableDataEditor& editor0 = schema.tableHandle(COND_AUTHENTICATION_TABLE).dataEditor();

    ora::SequenceManager sequenceMgr( SEQUENCE_TABLE_NAME,schema );
    principalId = sequenceMgr.getNextId( COND_AUTHENTICATION_TABLE, true );
        
    coral::AttributeList authData;
    editor0.rowBuffer(authData);
    authData[ PRINCIPAL_ID_COL ].data<int>() = principalId;
    authData[ PRINCIPAL_NAME_COL ].data<std::string>() = principalName;
    authData[ VERIFICATION_COL ].data<std::string>() = cipher.b64encrypt( principalName );
    authData[ PRINCIPAL_KEY_COL ].data<std::string>() = cipher.b64encrypt( principalKey );
    authData[ ADMIN_KEY_COL ].data<std::string>() = adminCipher.b64encrypt( principalKey );
    editor0.insertRow( authData );
  }

  if(setAdmin){
    std::string connString = m_serviceData->connectionString;
    std::string connLabel = schemaLabelForCredentialStore( connString );
    CredentialData credsData;
    bool found = selectConnection( schema, connLabel, credsData );
    if(!found){
      throwException("Credential Store connection has not been defined.","CredentialStore::updatePrincipal");
    }
    setPermission( principalId, principalKey, Auth::COND_ADMIN_ROLE, connString, credsData.id, adminCipher.b64decrypt( credsData.connectionKey ) );
  }

  session.close();
  return true;
}


bool cond::CredentialStore::setPermission( const std::string& principal, 
					   const std::string& role, 
					   const std::string& connectionString, 
					   const std::string& connectionLabel ){
  CSScopedSession session( *this );
  session.start( false  );

  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal( schema, principal, princData );

  if( ! found ){
    std::string msg = "Principal \"" + principal + "\" does not exist in the database.";
    throwException( msg, "CredentialStore::setPermission");
  }

  CredentialData credsData;
  found = selectConnection( schema, connectionLabel, credsData );
  
  if( ! found ){
    std::string msg = "Connection named \"" + connectionLabel + "\" does not exist in the database.";
    throwException( msg, "CredentialStore::setPermission");
  }

  Cipher cipher( m_principalKey );
  bool ret = setPermission( princData.id, cipher.b64decrypt( princData.adminKey), role, connectionString, credsData.id, cipher.b64decrypt( credsData.connectionKey ) );
  session.close();
  return ret;
}

bool cond::CredentialStore::unsetPermission( const std::string& principal, 
					     const std::string& role, 
					     const std::string& connectionString ){
  CSScopedSession session( *this );
  session.start( false  );
  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal( schema, principal, princData );

  if( ! found ){
    std::string msg = "Principal \"" + principal + "\" does not exist in the database.";
    throwException( msg, "CredentialStore::unsetPermission");
  }

  coral::ITableDataEditor& editor = schema.tableHandle(COND_AUTHORIZATION_TABLE).dataEditor();
  coral::AttributeList deleteData;
  deleteData.extend<int>( P_ID_COL );
  deleteData.extend<std::string>( ROLE_COL );
  deleteData.extend<std::string>( SCHEMA_COL );
  deleteData[ P_ID_COL ].data<int>() = princData.id;
  deleteData[ ROLE_COL ].data<std::string>() = role;
  deleteData[ SCHEMA_COL ].data<std::string>() = connectionString;
  std::stringstream whereClause;
  whereClause << P_ID_COL+" = :"+P_ID_COL;
  whereClause << " AND "<< ROLE_COL <<" = :"<<ROLE_COL;
  whereClause << " AND "<< SCHEMA_COL <<" = :"<<SCHEMA_COL;
  editor.deleteRows( whereClause.str(), deleteData );
  session.close();
  return true;
}

bool cond::CredentialStore::updateConnection( const std::string& connectionLabel, 
					      const std::string& userName, 
					      const std::string& password ){
  CSScopedSession session( *this );
  session.start( false  );

  m_session->transaction().start();

  updateConnection( connectionLabel,userName, password, true ); 

  session.close();
  return true;
}

bool cond::CredentialStore::removePrincipal( const std::string& principal ){
  CSScopedSession session( *this );
  session.start( false  );
  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal( schema, principal, princData );

  if( ! found ){
    std::string msg = "Principal \"" + principal + "\" does not exist in the database.";
    throwException( msg, "CredentialStore::removePrincipal");
  }

  coral::ITableDataEditor& editor0 = schema.tableHandle(COND_AUTHORIZATION_TABLE).dataEditor();

  coral::AttributeList deleteData0;
  deleteData0.extend<int>( P_ID_COL );
  deleteData0[ P_ID_COL ].data<int>() = princData.id;
  std::string whereClause0 = P_ID_COL+" = :"+P_ID_COL;
  editor0.deleteRows( whereClause0 , deleteData0 );

  coral::ITableDataEditor& editor1 = schema.tableHandle(COND_AUTHENTICATION_TABLE).dataEditor();

  coral::AttributeList deleteData1;
  deleteData1.extend<int>( PRINCIPAL_ID_COL );
  deleteData1[ PRINCIPAL_ID_COL ].data<int>() = princData.id;
  std::string whereClause1 = PRINCIPAL_ID_COL+" = :"+PRINCIPAL_ID_COL;
  editor1.deleteRows( whereClause1 , deleteData1 );

  session.close();
  
  return true;
}

bool cond::CredentialStore::removeConnection( const std::string& connectionLabel ){
  CSScopedSession session( *this );
  session.start( false  );
  coral::ISchema& schema = m_session->nominalSchema();

  CredentialData credsData;
  bool found = selectConnection( schema, connectionLabel, credsData );

  if( ! found ){
    std::string msg = "Connection named \"" + connectionLabel + "\" does not exist in the database.";
    throwException( msg, "CredentialStore::removeConnection");
  }

  coral::ITableDataEditor& editor0 = schema.tableHandle(COND_AUTHORIZATION_TABLE).dataEditor();

  coral::AttributeList deleteData0;
  deleteData0.extend<int>( C_ID_COL );
  deleteData0[ C_ID_COL ].data<int>() = credsData.id;
  std::string whereClause0 = C_ID_COL+" = :"+C_ID_COL;
  editor0.deleteRows( whereClause0 , deleteData0 );

  coral::ITableDataEditor& editor1 = schema.tableHandle(COND_CREDENTIAL_TABLE).dataEditor();

  coral::AttributeList deleteData1;
  deleteData1.extend<int>( CONNECTION_ID_COL );
  deleteData1[ CONNECTION_ID_COL ].data<int>() = credsData.id;
  std::string whereClause1 = CONNECTION_ID_COL+" = :"+CONNECTION_ID_COL;
  editor1.deleteRows( whereClause1 , deleteData1 );

  session.close();

  return true;
}

bool cond::CredentialStore::selectForUser( coral_bridge::AuthenticationCredentialSet& destinationData ){
  CSScopedSession session( *this );
  session.start( true  );
  coral::ISchema& schema = m_session->nominalSchema();

  Cipher cipher( m_principalKey );

  std::auto_ptr<coral::IQuery> query(schema.newQuery());
  query->addToTableList(COND_AUTHORIZATION_TABLE, "AUTHO");
  query->addToTableList(COND_CREDENTIAL_TABLE, "CREDS");
  coral::AttributeList readBuff;
  readBuff.extend<std::string>("AUTHO."+ROLE_COL);
  readBuff.extend<std::string>("AUTHO."+SCHEMA_COL);
  readBuff.extend<std::string>("AUTHO."+AUTH_KEY_COL);
  readBuff.extend<std::string>("CREDS."+CONNECTION_LABEL_COL);
  readBuff.extend<std::string>("CREDS."+USERNAME_COL);
  readBuff.extend<std::string>("CREDS."+PASSWORD_COL);
  readBuff.extend<std::string>("CREDS."+VERIFICATION_KEY_COL);
  coral::AttributeList whereData;
  whereData.extend<int>(P_ID_COL);
  whereData[ P_ID_COL ].data<int>() = m_principalId;
  std::stringstream whereClause;
  whereClause << "AUTHO."<< C_ID_COL << "="<<"CREDS."<<CONNECTION_ID_COL;
  whereClause << " AND " << "AUTHO."<< P_ID_COL << " = :"<<P_ID_COL;
  query->defineOutput(readBuff);
  query->addToOutputList( "AUTHO."+ROLE_COL );
  query->addToOutputList( "AUTHO."+SCHEMA_COL );
  query->addToOutputList( "AUTHO."+AUTH_KEY_COL );
  query->addToOutputList( "CREDS."+CONNECTION_LABEL_COL );
  query->addToOutputList( "CREDS."+USERNAME_COL );
  query->addToOutputList( "CREDS."+PASSWORD_COL );
  query->addToOutputList( "CREDS."+VERIFICATION_KEY_COL );
  query->setCondition( whereClause.str(), whereData );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    const coral::AttributeList& row = cursor.currentRow();
    const std::string& role = row[ "AUTHO."+ROLE_COL ].data<std::string>();
    const std::string& connectionString = row[ "AUTHO."+SCHEMA_COL ].data<std::string>();
    const std::string& encryptedAuthKey = row[ "AUTHO."+AUTH_KEY_COL ].data<std::string>();
    const std::string& connectionLabel = row[ "CREDS."+CONNECTION_LABEL_COL ].data<std::string>();
    const std::string& encryptedUserName = row[ "CREDS."+USERNAME_COL ].data<std::string>();
    const std::string& encryptedPassword = row[ "CREDS."+PASSWORD_COL ].data<std::string>();
    const std::string& encryptedLabel = row[ "CREDS."+VERIFICATION_KEY_COL ].data<std::string>();
    std::string authKey = cipher.b64decrypt( encryptedAuthKey );
    Cipher connCipher( authKey );
    if( connCipher.b64decrypt( encryptedLabel ) == connectionLabel ){
      destinationData.registerCredentials( connectionString, role, connCipher.b64decrypt( encryptedUserName ),  connCipher.b64decrypt( encryptedPassword ) );
    } 
  }
  session.close();
  return true;
}

bool cond::CredentialStore::importForPrincipal( const std::string& principal, 
						const coral_bridge::AuthenticationCredentialSet& dataSource,
						bool forceUpdateConnection ){
  CSScopedSession session( *this );
  session.start( false  );
  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal( schema, principal, princData );

  if( ! found ){
    std::string msg = "Principal \"" + principal + "\" does not exist in the database.";
    throwException( msg, "CredentialStore::importForPrincipal");
  }

  bool imported = false;
  Cipher cipher( m_principalKey );
  std::string princKey = cipher.b64decrypt( princData.adminKey);

  const std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >& creds = dataSource.data();
  for( std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >::const_iterator iConn = creds.begin(); iConn != creds.end(); ++iConn ){
    const std::string& connectionString = iConn->first.first;
    coral::URIParser parser;
    parser.setURI( connectionString );
    std::string serviceName = parser.hostName();
    const std::string& role = iConn->first.second;
    std::string userName = iConn->second->valueForItem( coral::IAuthenticationCredentials::userItem() );
    std::string password = iConn->second->valueForItem( coral::IAuthenticationCredentials::passwordItem());
    // first import the connections
    std::pair<int,std::string> conn = updateConnection( schemaLabel( serviceName, userName ), userName, password, forceUpdateConnection );
    Cipher cipher( m_principalKey );
    // than set the permission for the specific role
    setPermission( princData.id, princKey, role, connectionString, conn.first, conn.second );
    imported = true;
  }
  session.close();  
  return imported;
}

bool cond::CredentialStore::listPrincipals( std::vector<std::string>& destination ){

  CSScopedSession session( *this );
  session.start( true  );
  coral::ISchema& schema = m_session->nominalSchema();

  std::auto_ptr<coral::IQuery> query(schema.tableHandle(COND_AUTHENTICATION_TABLE).newQuery());
  coral::AttributeList readBuff;
  readBuff.extend<std::string>(PRINCIPAL_NAME_COL);
  query->defineOutput(readBuff);
  query->addToOutputList( PRINCIPAL_NAME_COL );
  coral::ICursor& cursor = query->execute();
  bool found = false;
  while ( cursor.next() ) {
    found = true;
    const coral::AttributeList& row = cursor.currentRow();
    destination.push_back( row[ PRINCIPAL_NAME_COL ].data<std::string>() );
  }
  session.close();    
  return found;
}


bool cond::CredentialStore::listConnections( std::map<std::string,std::pair<std::string,std::string> >& destination ){
  CSScopedSession session( *this );
  session.start( true  );
  coral::ISchema& schema = m_session->nominalSchema();

  std::auto_ptr<coral::IQuery> query(schema.tableHandle(COND_CREDENTIAL_TABLE).newQuery());
  coral::AttributeList readBuff;
  readBuff.extend<std::string>( CONNECTION_LABEL_COL );
  readBuff.extend<std::string>( USERNAME_COL );
  readBuff.extend<std::string>( PASSWORD_COL );
  readBuff.extend<std::string>( VERIFICATION_KEY_COL );
  readBuff.extend<std::string>( CONNECTION_KEY_COL );
  query->defineOutput(readBuff);
  query->addToOutputList( CONNECTION_LABEL_COL );
  query->addToOutputList( USERNAME_COL );
  query->addToOutputList( PASSWORD_COL );
  query->addToOutputList( VERIFICATION_KEY_COL );
  query->addToOutputList( CONNECTION_KEY_COL );
  coral::ICursor& cursor = query->execute();
  bool found = false;
  Cipher cipher0(m_principalKey );
  while ( cursor.next() ) {
    std::string userName("");
    std::string password("");
    const coral::AttributeList& row = cursor.currentRow();
    const std::string& connLabel = row[ CONNECTION_LABEL_COL].data<std::string>();
    const std::string& encryptedKey = row[ CONNECTION_KEY_COL].data<std::string>();
    const std::string& encryptedVerif = row[ VERIFICATION_KEY_COL].data<std::string>();
    std::string connKey = cipher0.b64decrypt( encryptedKey );
    Cipher cipher1( connKey );
    std::string verif = cipher1.b64decrypt( encryptedVerif );
    if( verif == connLabel ){
      const std::string& encryptedUserName = row[ USERNAME_COL].data<std::string>();
      const std::string& encryptedPassword = row[ PASSWORD_COL].data<std::string>();
      userName = cipher1.b64decrypt( encryptedUserName );
      password = cipher1.b64decrypt( encryptedPassword );
    }
    destination.insert( std::make_pair( connLabel, std::make_pair( userName, password ) ) );
    found = true;
  }
  session.close();    
  return found;
}

bool cond::CredentialStore::selectPermissions( const std::string& principalName, 
					       const std::string& role, 
					       const std::string& connectionString, 
					       std::vector<Permission>& destination ){
  CSScopedSession session( *this );
  session.start( true  );
  coral::ISchema& schema = m_session->nominalSchema();
  std::auto_ptr<coral::IQuery> query(schema.newQuery());
  query->addToTableList(COND_AUTHENTICATION_TABLE, "AUTHE");
  query->addToTableList(COND_AUTHORIZATION_TABLE, "AUTHO");
  query->addToTableList(COND_CREDENTIAL_TABLE, "CREDS");
  coral::AttributeList readBuff;
  readBuff.extend<std::string>("AUTHE."+PRINCIPAL_NAME_COL);
  readBuff.extend<std::string>("AUTHO."+ROLE_COL);
  readBuff.extend<std::string>("AUTHO."+SCHEMA_COL);
  readBuff.extend<std::string>("CREDS."+CONNECTION_LABEL_COL);
  coral::AttributeList whereData;
  std::stringstream whereClause;
  whereClause << "AUTHE."<< PRINCIPAL_ID_COL << "= AUTHO."<< P_ID_COL;
  whereClause << " AND AUTHO."<< C_ID_COL << "="<<"CREDS."<<CONNECTION_ID_COL;
  if( !principalName.empty() ){
    whereData.extend<std::string>(PRINCIPAL_NAME_COL);
    whereData[ PRINCIPAL_NAME_COL ].data<std::string>() = principalName;
    whereClause << " AND AUTHE."<< PRINCIPAL_NAME_COL <<" = :"<<PRINCIPAL_NAME_COL;
  }
  if( !role.empty() ){
    whereData.extend<std::string>(ROLE_COL);
    whereData[ ROLE_COL ].data<std::string>() = role;
    whereClause << " AND AUTHO."<< ROLE_COL <<" = :"<<ROLE_COL;
  }
  if( !connectionString.empty() ){
    whereData.extend<std::string>(SCHEMA_COL);
    whereData[ SCHEMA_COL ].data<std::string>() = connectionString;
    whereClause << " AND AUTHO."<< SCHEMA_COL <<" = :"<<SCHEMA_COL;
  }
  
  query->defineOutput(readBuff);
  query->addToOutputList( "AUTHE."+PRINCIPAL_NAME_COL );
  query->addToOutputList( "AUTHO."+ROLE_COL );
  query->addToOutputList( "AUTHO."+SCHEMA_COL );
  query->addToOutputList( "CREDS."+CONNECTION_LABEL_COL );
  query->setCondition( whereClause.str(), whereData );
  query->addToOrderList( "AUTHO."+SCHEMA_COL );
  query->addToOrderList( "AUTHE."+PRINCIPAL_NAME_COL  );
  query->addToOrderList( "AUTHO."+ROLE_COL  );
  coral::ICursor& cursor = query->execute();
  bool found = false;
  while ( cursor.next() ) {
    const coral::AttributeList& row = cursor.currentRow();
    destination.resize( destination.size()+1 );
    Permission& perm = destination.back();
    perm.principalName = row[ "AUTHE."+PRINCIPAL_NAME_COL ].data<std::string>();
    perm.role = row[ "AUTHO."+ROLE_COL ].data<std::string>();
    perm.connectionString = row[ "AUTHO."+SCHEMA_COL ].data<std::string>();
    perm.connectionLabel = row[ "CREDS."+CONNECTION_LABEL_COL ].data<std::string>();
    found = true;
  }
  session.close();
  return found;  
}

bool cond::CredentialStore::exportAll( coral_bridge::AuthenticationCredentialSet& data ){
  CSScopedSession session( *this );
  session.start( true  );
  coral::ISchema& schema = m_session->nominalSchema();
  std::auto_ptr<coral::IQuery> query(schema.newQuery());
  query->addToTableList(COND_AUTHORIZATION_TABLE, "AUTHO");
  query->addToTableList(COND_CREDENTIAL_TABLE, "CREDS");
  coral::AttributeList readBuff;
  readBuff.extend<std::string>("AUTHO."+ROLE_COL);
  readBuff.extend<std::string>("AUTHO."+SCHEMA_COL);
  readBuff.extend<std::string>("CREDS."+CONNECTION_LABEL_COL);
  readBuff.extend<std::string>("CREDS."+VERIFICATION_KEY_COL);
  readBuff.extend<std::string>("CREDS."+CONNECTION_KEY_COL);
  readBuff.extend<std::string>("CREDS."+USERNAME_COL);
  readBuff.extend<std::string>("CREDS."+PASSWORD_COL);
  coral::AttributeList whereData;
  std::stringstream whereClause;
  whereClause << "AUTHO."<< C_ID_COL << "="<<"CREDS."<<CONNECTION_ID_COL;
  
  query->defineOutput(readBuff);
  query->addToOutputList( "AUTHO."+ROLE_COL );
  query->addToOutputList( "AUTHO."+SCHEMA_COL );
  query->addToOutputList( "CREDS."+CONNECTION_LABEL_COL );
  query->addToOutputList( "CREDS."+VERIFICATION_KEY_COL );
  query->addToOutputList( "CREDS."+CONNECTION_KEY_COL );
  query->addToOutputList( "CREDS."+USERNAME_COL );
  query->addToOutputList( "CREDS."+PASSWORD_COL );
  query->setCondition( whereClause.str(), whereData );
  coral::ICursor& cursor = query->execute();
  bool found = false;
  Cipher cipher0( m_principalKey );
  while ( cursor.next() ) {
    const coral::AttributeList& row = cursor.currentRow();
    const std::string& role = row[ "AUTHO."+ROLE_COL ].data<std::string>();
    const std::string& connectionString = row[ "AUTHO."+SCHEMA_COL ].data<std::string>();
    const std::string& connectionLabel = row[ "CREDS."+CONNECTION_LABEL_COL ].data<std::string>();
    const std::string& encryptedVerifKey = row[ "CREDS."+VERIFICATION_KEY_COL ].data<std::string>();
    const std::string& encryptedConnection = row[ "CREDS."+CONNECTION_KEY_COL ].data<std::string>();
    std::string userName("");
    std::string password("");
    std::string connectionKey = cipher0.b64decrypt( encryptedConnection );
    Cipher cipher1( connectionKey );
    std::string verifKey = cipher1.b64decrypt( encryptedVerifKey );
    if( verifKey == connectionLabel ){
      const std::string& encryptedUserName = row[ "CREDS."+USERNAME_COL].data<std::string>();
      const std::string& encryptedPassword = row[ "CREDS."+PASSWORD_COL].data<std::string>();
      userName = cipher1.b64decrypt( encryptedUserName );
      password = cipher1.b64decrypt( encryptedPassword );
    }
    data.registerCredentials( connectionString, role, userName, password );
    found = true;
  }
  session.close();
  return found;  
}

const std::string& cond::CredentialStore::keyPrincipalName (){
  return m_key.principalName();
}


