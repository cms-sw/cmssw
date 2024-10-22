#include "CondCore/CondDB/interface/CredentialStore.h"
#include "CondCore/CondDB/interface/Cipher.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Auth.h"
//
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
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
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
//
#include "RelationalAccess/AuthenticationCredentials.h"
//
#include <filesystem>
#include <fstream>
#include <sstream>

static const std::string serviceName = "CondAuthenticationService";

coral_bridge::AuthenticationCredentialSet::AuthenticationCredentialSet() : m_data() {}

coral_bridge::AuthenticationCredentialSet::~AuthenticationCredentialSet() { reset(); }

void coral_bridge::AuthenticationCredentialSet::reset() {
  for (auto iData = m_data.begin(); iData != m_data.end(); ++iData)
    delete iData->second;
  m_data.clear();
}

void coral_bridge::AuthenticationCredentialSet::registerItem(const std::string& connectionString,
                                                             const std::string& itemName,
                                                             const std::string& itemValue) {
  registerItem(connectionString, cond::auth::COND_DEFAULT_ROLE, itemName, itemValue);
}

void coral_bridge::AuthenticationCredentialSet::registerItem(const std::string& connectionString,
                                                             const std::string& role,
                                                             const std::string& itemName,
                                                             const std::string& itemValue) {
  std::pair<std::string, std::string> connKey(connectionString, role);
  std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>::iterator iData =
      m_data.find(connKey);
  if (iData == m_data.end()) {
    iData = m_data.insert(std::make_pair(connKey, new coral::AuthenticationCredentials(serviceName))).first;
  }
  iData = m_data.insert(std::make_pair(connKey, new coral::AuthenticationCredentials(serviceName))).first;
  iData->second->registerItem(itemName, itemValue);
}

void coral_bridge::AuthenticationCredentialSet::registerCredentials(const std::string& connectionString,
                                                                    const std::string& userName,
                                                                    const std::string& password) {
  registerCredentials(connectionString, cond::auth::COND_DEFAULT_ROLE, userName, password);
}

void coral_bridge::AuthenticationCredentialSet::registerCredentials(const std::string& connectionString,
                                                                    const std::string& role,
                                                                    const std::string& userName,
                                                                    const std::string& password) {
  std::pair<std::string, std::string> connKey(connectionString, role);
  std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>::iterator iData =
      m_data.find(connKey);
  if (iData != m_data.end()) {
    delete iData->second;
    m_data.erase(connKey);
  }
  iData = m_data.insert(std::make_pair(connKey, new coral::AuthenticationCredentials(serviceName))).first;
  iData->second->registerItem(coral::IAuthenticationCredentials::userItem(), userName);
  iData->second->registerItem(coral::IAuthenticationCredentials::passwordItem(), password);
}

void coral_bridge::AuthenticationCredentialSet::import(const AuthenticationCredentialSet& data) {
  for (std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>::const_iterator iData =
           data.m_data.begin();
       iData != data.m_data.end();
       ++iData) {
    registerCredentials(iData->first.first,
                        iData->first.second,
                        iData->second->valueForItem(coral::IAuthenticationCredentials::userItem()),
                        iData->second->valueForItem(coral::IAuthenticationCredentials::passwordItem()));
  }
}

const coral::IAuthenticationCredentials* coral_bridge::AuthenticationCredentialSet::get(
    const std::string& connectionString) const {
  return get(connectionString, cond::auth::COND_DEFAULT_ROLE);
}

const coral::IAuthenticationCredentials* coral_bridge::AuthenticationCredentialSet::get(
    const std::string& connectionString, const std::string& role) const {
  const coral::IAuthenticationCredentials* ret = nullptr;
  std::pair<std::string, std::string> connKey(connectionString, role);
  std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>::const_iterator iData =
      m_data.find(connKey);
  if (iData != m_data.end()) {
    ret = iData->second;
  }
  return ret;
}

const std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>&
coral_bridge::AuthenticationCredentialSet::data() const {
  return m_data;
}

static const std::string TABLE_PREFIX("DB_");
static const std::string LEGACY_TABLE_PREFIX("COND_");
static const std::string SEQUENCE_TABLE("CREDENTIAL_SEQUENCE");
static const std::string SEQUENCE_NAME_COL("NAME");
static const std::string SEQUENCE_VALUE_COL("VALUE");

static const std::string AUTHENTICATION_TABLE("AUTHENTICATION");
static const std::string PRINCIPAL_ID_COL("P_ID");
static const std::string PRINCIPAL_NAME_COL("P_NAME");
static const std::string VERIFICATION_COL("CRED0");
static const std::string PRINCIPAL_KEY_COL("CRED1");
static const std::string ADMIN_KEY_COL("CRED2");

static const std::string AUTHORIZATION_TABLE("AUTHORIZATION");
static const std::string AUTH_ID_COL("AUTH_ID");
static const std::string P_ID_COL("P_ID");
static const std::string ROLE_COL("C_ROLE");
static const std::string SCHEMA_COL("C_SCHEMA");
static const std::string AUTH_KEY_COL("CRED3");
static const std::string C_ID_COL("C_ID");

static const std::string CREDENTIAL_TABLE("CREDENTIAL");
static const std::string CONNECTION_ID_COL("CONN_ID");
static const std::string CONNECTION_LABEL_COL("CONN_LABEL");
static const std::string USERNAME_COL("CRED4");
static const std::string PASSWORD_COL("CRED5");
static const std::string VERIFICATION_KEY_COL("CRED6");
static const std::string CONNECTION_KEY_COL("CRED7");

const std::string DEFAULT_DATA_SOURCE("Cond_Default_Authentication");

std::string tname(const std::string& tableName, const std::string& schemaVersion) {
  std::string prefix(TABLE_PREFIX);
  if (schemaVersion.empty())
    prefix = LEGACY_TABLE_PREFIX;
  return prefix + tableName;
}

// implementation functions and helpers
namespace cond {

  std::string schemaLabel(const std::string& serviceName, const std::string& userName) {
    std::string ret = userName;
    if (!serviceName.empty()) {
      ret += "@" + serviceName;
      ret = to_lower(ret);
    }
    return ret;
  }

  std::string schemaLabelForCredentialStore(const std::string& connectionString) {
    coral::URIParser parser;
    parser.setURI(connectionString);
    std::string serviceName = parser.hostName();
    std::string schemaName = parser.databaseOrSchemaName();
    return schemaLabel(serviceName, schemaName);
  }

  class CSScopedSession {
  public:
    CSScopedSession(CredentialStore& store) : m_store(store) {}
    ~CSScopedSession() { m_store.closeSession(false); }
    void startSuper(const std::string& connectionString, const std::string& userName, const std::string& password) {
      m_store.startSuperSession(connectionString, userName, password);
    }
    void start(bool readOnly = true) { m_store.startSession(readOnly); }
    void close() { m_store.closeSession(); }

  private:
    CredentialStore& m_store;
  };

  struct PrincipalData {
    int id;
    std::string verifKey;
    std::string principalKey;
    std::string adminKey;
    PrincipalData() : id(-1), verifKey(""), principalKey(""), adminKey("") {}
  };
  bool selectPrincipal(const std::string& schemaVersion,
                       coral::ISchema& schema,
                       const std::string& principal,
                       PrincipalData& destination) {
    std::unique_ptr<coral::IQuery> query(schema.tableHandle(tname(AUTHENTICATION_TABLE, schemaVersion)).newQuery());
    coral::AttributeList readBuff;
    readBuff.extend<int>(PRINCIPAL_ID_COL);
    readBuff.extend<std::string>(VERIFICATION_COL);
    readBuff.extend<std::string>(PRINCIPAL_KEY_COL);
    readBuff.extend<std::string>(ADMIN_KEY_COL);
    coral::AttributeList whereData;
    whereData.extend<std::string>(PRINCIPAL_NAME_COL);
    whereData[PRINCIPAL_NAME_COL].data<std::string>() = principal;
    std::string whereClause = PRINCIPAL_NAME_COL + " = :" + PRINCIPAL_NAME_COL;
    query->defineOutput(readBuff);
    query->addToOutputList(PRINCIPAL_ID_COL);
    query->addToOutputList(VERIFICATION_COL);
    query->addToOutputList(PRINCIPAL_KEY_COL);
    query->addToOutputList(ADMIN_KEY_COL);
    query->setCondition(whereClause, whereData);
    coral::ICursor& cursor = query->execute();
    bool found = false;
    if (cursor.next()) {
      found = true;
      const coral::AttributeList& row = cursor.currentRow();
      destination.id = row[PRINCIPAL_ID_COL].data<int>();
      destination.verifKey = row[VERIFICATION_COL].data<std::string>();
      destination.principalKey = row[PRINCIPAL_KEY_COL].data<std::string>();
      destination.adminKey = row[ADMIN_KEY_COL].data<std::string>();
    }
    return found;
  }

  struct CredentialData {
    int id;
    std::string userName;
    std::string password;
    std::string connectionKey;
    std::string verificationKey;
    CredentialData() : id(-1), userName(""), password(""), connectionKey("") {}
  };

  bool selectConnection(const std::string& schemaVersion,
                        coral::ISchema& schema,
                        const std::string& connectionLabel,
                        CredentialData& destination) {
    std::unique_ptr<coral::IQuery> query(schema.tableHandle(tname(CREDENTIAL_TABLE, schemaVersion)).newQuery());
    coral::AttributeList readBuff;
    readBuff.extend<int>(CONNECTION_ID_COL);
    readBuff.extend<std::string>(USERNAME_COL);
    readBuff.extend<std::string>(PASSWORD_COL);
    readBuff.extend<std::string>(VERIFICATION_KEY_COL);
    readBuff.extend<std::string>(CONNECTION_KEY_COL);
    coral::AttributeList whereData;
    whereData.extend<std::string>(CONNECTION_LABEL_COL);
    whereData[CONNECTION_LABEL_COL].data<std::string>() = connectionLabel;
    std::string whereClause = CONNECTION_LABEL_COL + " = :" + CONNECTION_LABEL_COL;
    query->defineOutput(readBuff);
    query->addToOutputList(CONNECTION_ID_COL);
    query->addToOutputList(USERNAME_COL);
    query->addToOutputList(PASSWORD_COL);
    query->addToOutputList(VERIFICATION_KEY_COL);
    query->addToOutputList(CONNECTION_KEY_COL);
    query->setCondition(whereClause, whereData);
    coral::ICursor& cursor = query->execute();
    bool found = false;
    if (cursor.next()) {
      const coral::AttributeList& row = cursor.currentRow();
      destination.id = row[CONNECTION_ID_COL].data<int>();
      destination.userName = row[USERNAME_COL].data<std::string>();
      destination.password = row[PASSWORD_COL].data<std::string>();
      destination.verificationKey = row[VERIFICATION_KEY_COL].data<std::string>();
      destination.connectionKey = row[CONNECTION_KEY_COL].data<std::string>();
      found = true;
    }
    return found;
  }

  struct AuthorizationData {
    int id;
    int connectionId;
    std::string key;
    AuthorizationData() : id(-1), connectionId(-1), key("") {}
  };

  bool selectAuthorization(const std::string& schemaVersion,
                           coral::ISchema& schema,
                           int principalId,
                           const std::string& role,
                           const std::string& connectionString,
                           AuthorizationData& destination) {
    std::unique_ptr<coral::IQuery> query(schema.tableHandle(tname(AUTHORIZATION_TABLE, schemaVersion)).newQuery());
    coral::AttributeList readBuff;
    readBuff.extend<int>(AUTH_ID_COL);
    readBuff.extend<int>(C_ID_COL);
    readBuff.extend<std::string>(AUTH_KEY_COL);
    coral::AttributeList whereData;
    whereData.extend<int>(P_ID_COL);
    whereData.extend<std::string>(ROLE_COL);
    whereData.extend<std::string>(SCHEMA_COL);
    whereData[P_ID_COL].data<int>() = principalId;
    whereData[ROLE_COL].data<std::string>() = role;
    whereData[SCHEMA_COL].data<std::string>() = to_lower(connectionString);
    std::stringstream whereClause;
    whereClause << P_ID_COL << " = :" << P_ID_COL;
    whereClause << " AND " << ROLE_COL << " = :" << ROLE_COL;
    whereClause << " AND " << SCHEMA_COL << " = :" << SCHEMA_COL;
    query->defineOutput(readBuff);
    query->addToOutputList(AUTH_ID_COL);
    query->addToOutputList(C_ID_COL);
    query->addToOutputList(AUTH_KEY_COL);
    query->setCondition(whereClause.str(), whereData);
    coral::ICursor& cursor = query->execute();
    bool found = false;
    if (cursor.next()) {
      const coral::AttributeList& row = cursor.currentRow();
      destination.id = row[AUTH_ID_COL].data<int>();
      destination.connectionId = row[C_ID_COL].data<int>();
      destination.key = row[AUTH_KEY_COL].data<std::string>();
      found = true;
    }
    return found;
  }

  size_t getAuthorizationEntries(const std::string& schemaVersion,
                                 coral::ISchema& schema,
                                 int principalId,
                                 const std::string& role,
                                 const std::string& connectionString) {
    std::unique_ptr<coral::IQuery> query(schema.tableHandle(tname(AUTHORIZATION_TABLE, schemaVersion)).newQuery());
    coral::AttributeList readBuff;
    readBuff.extend<int>(AUTH_ID_COL);
    coral::AttributeList whereData;
    whereData.extend<std::string>(SCHEMA_COL);
    std::stringstream whereClause;
    whereClause << SCHEMA_COL << " = :" << SCHEMA_COL;
    if (principalId >= 0) {
      whereData.extend<int>(P_ID_COL);
      whereClause << "AND" << P_ID_COL << " = :" << P_ID_COL;
    }
    if (!role.empty()) {
      whereData.extend<std::string>(ROLE_COL);
      whereClause << " AND " << ROLE_COL << " = :" << ROLE_COL;
    }
    whereData[SCHEMA_COL].data<std::string>() = connectionString;
    if (principalId >= 0)
      whereData[P_ID_COL].data<int>() = principalId;
    if (!role.empty())
      whereData[ROLE_COL].data<std::string>() = role;
    query->defineOutput(readBuff);
    query->addToOutputList(AUTH_ID_COL);
    query->setCondition(whereClause.str(), whereData);
    coral::ICursor& cursor = query->execute();
    size_t n_entries = 0;
    while (cursor.next()) {
      n_entries += 1;
    }
    return n_entries;
  }

  bool getNextSequenceValue(const std::string& schemaVersion,
                            coral::ISchema& schema,
                            const std::string& sequenceName,
                            int& value) {
    bool ret = false;
    std::unique_ptr<coral::IQuery> query(schema.tableHandle(tname(SEQUENCE_TABLE, schemaVersion)).newQuery());
    query->limitReturnedRows(1, 0);
    query->addToOutputList(SEQUENCE_VALUE_COL);
    query->defineOutputType(SEQUENCE_VALUE_COL, coral::AttributeSpecification::typeNameForType<int>());
    query->setForUpdate();
    std::string whereClause(SEQUENCE_NAME_COL + " = :" + SEQUENCE_NAME_COL);
    coral::AttributeList rowData;
    rowData.extend<std::string>(SEQUENCE_NAME_COL);
    rowData.begin()->data<std::string>() = sequenceName;
    query->setCondition(whereClause, rowData);
    coral::ICursor& cursor = query->execute();
    if (cursor.next()) {
      value = cursor.currentRow().begin()->data<int>() + 1;
      ret = true;
    } else {
      return false;
    }
    // update...
    coral::AttributeList updateData;
    updateData.extend<std::string>(SEQUENCE_NAME_COL);
    updateData.extend<int>(SEQUENCE_VALUE_COL);
    std::string setClause(SEQUENCE_VALUE_COL + " = :" + SEQUENCE_VALUE_COL);
    std::string whClause(SEQUENCE_NAME_COL + " = :" + SEQUENCE_NAME_COL);
    coral::AttributeList::iterator iAttribute = updateData.begin();
    iAttribute->data<std::string>() = sequenceName;
    ++iAttribute;
    iAttribute->data<int>() = value;
    schema.tableHandle(tname(SEQUENCE_TABLE, schemaVersion)).dataEditor().updateRows(setClause, whClause, updateData);
    return ret;
  }

  std::pair<int, std::string> updatePrincipalData(const std::string& schemaVersion,
                                                  coral::ISchema& schema,
                                                  const std::string& authenticationKey,
                                                  const std::string& principalName,
                                                  const std::string& adminKey,
                                                  bool init /**= false **/,
                                                  std::stringstream& log) {
    PrincipalData princData;
    bool found = selectPrincipal(schemaVersion, schema, principalName, princData);

    auth::Cipher cipher0(authenticationKey);
    auth::Cipher cipher1(adminKey);

    std::string verifStr = cipher0.b64encrypt(principalName);
    std::string principalKey("");
    int principalId = princData.id;

    std::string authentication_table_name = tname(AUTHENTICATION_TABLE, schemaVersion);

    coral::ITableDataEditor& editor = schema.tableHandle(authentication_table_name).dataEditor();
    if (found) {
      log << "Updating existing principal " << principalName << " (id: " << principalId << " )" << std::endl;
      principalKey = cipher1.b64decrypt(princData.adminKey);
      coral::AttributeList updateData;
      updateData.extend<int>(PRINCIPAL_ID_COL);
      updateData.extend<std::string>(VERIFICATION_COL);
      updateData.extend<std::string>(PRINCIPAL_KEY_COL);
      updateData.extend<std::string>(ADMIN_KEY_COL);
      updateData[PRINCIPAL_ID_COL].data<int>() = principalId;
      updateData[VERIFICATION_COL].data<std::string>() = verifStr;
      updateData[PRINCIPAL_KEY_COL].data<std::string>() = cipher0.b64encrypt(principalKey);
      updateData[ADMIN_KEY_COL].data<std::string>() = cipher1.b64encrypt(principalKey);
      std::stringstream setClause;
      setClause << VERIFICATION_COL << " = :" << VERIFICATION_COL << ", ";
      setClause << PRINCIPAL_KEY_COL << " = :" << PRINCIPAL_KEY_COL << ", ";
      setClause << ADMIN_KEY_COL << " = :" << ADMIN_KEY_COL;
      std::string whereClause = PRINCIPAL_ID_COL + " = :" + PRINCIPAL_ID_COL;
      editor.updateRows(setClause.str(), whereClause, updateData);
    } else {
      if (init) {
        principalKey = adminKey;
      } else {
        auth::KeyGenerator gen;
        principalKey = gen.make(auth::COND_DB_KEY_SIZE);
      }
      coral::ITableDataEditor& editor0 = schema.tableHandle(authentication_table_name).dataEditor();

      if (!getNextSequenceValue(schemaVersion, schema, authentication_table_name, principalId))
        throwException("Can't find " + authentication_table_name + " sequence.", "CredentialStore::updatePrincipal");
      log << "Creating new principal " << principalName << " (id: " << principalId << " )" << std::endl;
      coral::AttributeList authData;
      editor0.rowBuffer(authData);
      authData[PRINCIPAL_ID_COL].data<int>() = principalId;
      authData[PRINCIPAL_NAME_COL].data<std::string>() = principalName;
      authData[VERIFICATION_COL].data<std::string>() = verifStr;
      authData[PRINCIPAL_KEY_COL].data<std::string>() = cipher0.b64encrypt(principalKey);
      authData[ADMIN_KEY_COL].data<std::string>() = cipher1.b64encrypt(principalKey);
      editor0.insertRow(authData);
    }

    return std::make_pair(principalId, principalKey);
  }

  bool setPermissionData(const std::string& schemaVersion,
                         coral::ISchema& schema,
                         int principalId,
                         const std::string& principalKey,
                         const std::string& role,
                         const std::string& connectionString,
                         int connectionId,
                         const std::string& connectionKey,
                         std::stringstream& log) {
    if (cond::auth::ROLES.find(role) == cond::auth::ROLES.end()) {
      throwException(std::string("Role ") + role + " does not exists.", "CredentialStore::setPermission");
    }
    auth::Cipher cipher(principalKey);
    std::string encryptedConnectionKey = cipher.b64encrypt(connectionKey);
    AuthorizationData authData;
    bool found = selectAuthorization(schemaVersion, schema, principalId, role, connectionString, authData);

    std::string authorization_table_name = tname(AUTHORIZATION_TABLE, schemaVersion);
    coral::ITableDataEditor& editor = schema.tableHandle(authorization_table_name).dataEditor();
    if (found) {
      log << "Updating permission for principal id " << principalId << " to access resource " << connectionString
          << " with role " << role << std::endl;
      coral::AttributeList updateData;
      updateData.extend<int>(AUTH_ID_COL);
      updateData.extend<int>(C_ID_COL);
      updateData.extend<std::string>(AUTH_KEY_COL);
      updateData[AUTH_ID_COL].data<int>() = authData.id;
      updateData[C_ID_COL].data<int>() = connectionId;
      updateData[AUTH_KEY_COL].data<std::string>() = encryptedConnectionKey;
      std::string setCl = C_ID_COL + " = :" + C_ID_COL + ", " + AUTH_KEY_COL + " = :" + AUTH_KEY_COL;
      std::string whereCl = AUTH_ID_COL + " = :" + AUTH_ID_COL;
      editor.updateRows(setCl, whereCl, updateData);
    } else {
      int next = -1;
      if (!getNextSequenceValue(schemaVersion, schema, authorization_table_name, next))
        throwException("Can't find " + authorization_table_name + " sequence.", "CredentialStore::setPermission");
      log << "Setting permission for principal id " << principalId << " to access resource " << connectionString
          << " with role " << role << std::endl;
      coral::AttributeList insertData;
      insertData.extend<int>(AUTH_ID_COL);
      insertData.extend<int>(P_ID_COL);
      insertData.extend<std::string>(ROLE_COL);
      insertData.extend<std::string>(SCHEMA_COL);
      insertData.extend<std::string>(AUTH_KEY_COL);
      insertData.extend<int>(C_ID_COL);
      insertData[AUTH_ID_COL].data<int>() = next;
      insertData[P_ID_COL].data<int>() = principalId;
      insertData[ROLE_COL].data<std::string>() = role;
      insertData[SCHEMA_COL].data<std::string>() = to_lower(connectionString);
      insertData[AUTH_KEY_COL].data<std::string>() = encryptedConnectionKey;
      insertData[C_ID_COL].data<int>() = connectionId;
      editor.insertRow(insertData);
    }
    return true;
  }

  std::pair<int, std::string> updateConnectionData(const std::string& schemaVersion,
                                                   coral::ISchema& schema,
                                                   const std::string& adminKey,
                                                   const std::string& connectionLabel,
                                                   const std::string& userName,
                                                   const std::string& password,
                                                   bool forceUpdate,
                                                   std::stringstream& log) {
    CredentialData credsData;
    bool found = selectConnection(schemaVersion, schema, connectionLabel, credsData);
    int connId = credsData.id;

    auth::Cipher adminCipher(adminKey);
    std::string connectionKey("");
    std::string credential_table_name = tname(CREDENTIAL_TABLE, schemaVersion);
    coral::ITableDataEditor& editor = schema.tableHandle(credential_table_name).dataEditor();
    if (found) {
      connectionKey = adminCipher.b64decrypt(credsData.connectionKey);
      auth::Cipher cipher(connectionKey);
      std::string verificationKey = cipher.b64decrypt(credsData.verificationKey);
      if (verificationKey != connectionLabel) {
        throwException("Decoding of connection key failed.", "CredentialStore::updateConnection");
      }
      if (forceUpdate) {
        std::string encryptedUserName = cipher.b64encrypt(userName);
        std::string encryptedPassword = cipher.b64encrypt(password);
        log << "Forcing update of connection " << connectionLabel << std::endl;
        coral::AttributeList updateData;
        updateData.extend<int>(CONNECTION_ID_COL);
        updateData.extend<std::string>(USERNAME_COL);
        updateData.extend<std::string>(PASSWORD_COL);
        updateData[CONNECTION_ID_COL].data<int>() = connId;
        updateData[USERNAME_COL].data<std::string>() = encryptedUserName;
        updateData[PASSWORD_COL].data<std::string>() = encryptedPassword;
        std::stringstream setCl;
        setCl << USERNAME_COL << " = :" << USERNAME_COL;
        setCl << ", " << PASSWORD_COL << " = :" << PASSWORD_COL;
        std::string whereCl = CONNECTION_ID_COL + " = :" + CONNECTION_ID_COL;
        editor.updateRows(setCl.str(), whereCl, updateData);
      }
    } else {
      auth::KeyGenerator gen;
      connectionKey = gen.make(auth::COND_DB_KEY_SIZE);
      auth::Cipher cipher(connectionKey);
      std::string encryptedUserName = cipher.b64encrypt(userName);
      std::string encryptedPassword = cipher.b64encrypt(password);
      std::string encryptedLabel = cipher.b64encrypt(connectionLabel);

      if (!getNextSequenceValue(schemaVersion, schema, credential_table_name, connId))
        throwException("Can't find " + credential_table_name + " sequence.", "CredentialStore::updateConnection");
      log << "Creating new connection " << connectionLabel << std::endl;
      coral::AttributeList insertData;
      insertData.extend<int>(CONNECTION_ID_COL);
      insertData.extend<std::string>(CONNECTION_LABEL_COL);
      insertData.extend<std::string>(USERNAME_COL);
      insertData.extend<std::string>(PASSWORD_COL);
      insertData.extend<std::string>(VERIFICATION_KEY_COL);
      insertData.extend<std::string>(CONNECTION_KEY_COL);
      insertData[CONNECTION_ID_COL].data<int>() = connId;
      insertData[CONNECTION_LABEL_COL].data<std::string>() = connectionLabel;
      insertData[USERNAME_COL].data<std::string>() = encryptedUserName;
      insertData[PASSWORD_COL].data<std::string>() = encryptedPassword;
      insertData[VERIFICATION_KEY_COL].data<std::string>() = encryptedLabel;
      insertData[CONNECTION_KEY_COL].data<std::string>() = adminCipher.b64encrypt(connectionKey);
      ;
      editor.insertRow(insertData);
    }
    return std::make_pair(connId, connectionKey);
  }

}  // namespace cond

// class private methods
void cond::CredentialStore::closeSession(bool commit) {
  if (m_session.get()) {
    if (m_session->transaction().isActive()) {
      if (commit) {
        m_session->transaction().commit();
      } else {
        m_session->transaction().rollback();
      }
    }
    m_session->endUserSession();
  }
  m_session.reset();
  if (m_connection.get()) {
    m_connection->disconnect();
  }
  m_connection.reset();
  m_log << "Session has been closed." << std::endl;
}

std::pair<std::string, std::string> cond::CredentialStore::openConnection(const std::string& connectionString) {
  coral::IHandle<coral::IRelationalService> relationalService =
      coral::Context::instance().query<coral::IRelationalService>();
  if (!relationalService.isValid()) {
    coral::Context::instance().loadComponent("CORAL/Services/RelationalService");
    relationalService = coral::Context::instance().query<coral::IRelationalService>();
  }
  coral::IRelationalDomain& domain = relationalService->domainForConnection(connectionString);
  std::pair<std::string, std::string> connTokens = domain.decodeUserConnectionString(connectionString);
  m_connection.reset(domain.newConnection(connTokens.first));
  m_connection->connect();
  return connTokens;
}

void cond::CredentialStore::openSession(const std::string& schemaName,
                                        const std::string& userName,
                                        const std::string& password,
                                        bool readMode) {
  coral::AccessMode accessMode = coral::ReadOnly;
  if (!readMode)
    accessMode = coral::Update;
  m_session.reset(m_connection->newSession(schemaName, accessMode));
  m_session->startUserSession(userName, password);
  // open read-only transaction
  m_session->transaction().start(readMode);
  m_log << "New session opened." << std::endl;
}

void cond::CredentialStore::startSuperSession(const std::string& connectionString,
                                              const std::string& userName,
                                              const std::string& password) {
  std::pair<std::string, std::string> connTokens = openConnection(connectionString);
  openSession(connTokens.second, userName, password, false);
}

// open session on the storage
void cond::CredentialStore::startSession(bool readMode) {
  if (!m_serviceData) {
    throwException("The credential store has not been initialized.", "cond::CredentialStore::openConnection");
  }
  const std::string& storeConnectionString = m_serviceData->connectionString;

  std::pair<std::string, std::string> connTokens = openConnection(storeConnectionString);

  const std::string& userName = m_serviceData->userName;
  const std::string& password = m_serviceData->password;

  openSession(connTokens.second, userName, password, true);

  coral::ISchema& schema = m_session->nominalSchema();
  const std::string& schemaVersion = m_key.version();
  if (!schema.existsTable(tname(AUTHENTICATION_TABLE, schemaVersion)) ||
      !schema.existsTable(tname(AUTHORIZATION_TABLE, schemaVersion)) ||
      !schema.existsTable(tname(CREDENTIAL_TABLE, schemaVersion))) {
    throwException("Credential database does not exists in \"" + storeConnectionString + "\"",
                   "CredentialStore::startSession");
  }

  const std::string& principalName = m_key.principalName();
  // now authenticate...
  PrincipalData princData;
  if (!selectPrincipal(schemaVersion, m_session->nominalSchema(), principalName, princData)) {
    throwException("Invalid credentials provided.(0)", "CredentialStore::startSession");
  }
  auth::Cipher cipher0(m_key.principalKey());
  std::string verifStr = cipher0.b64decrypt(princData.verifKey);
  if (verifStr != principalName) {
    throwException("Invalid credentials provided (1)", "CredentialStore::startSession");
  }
  // ok, authenticated!
  m_principalId = princData.id;
  m_principalKey = cipher0.b64decrypt(princData.principalKey);
  m_authenticatedPrincipal = m_key.principalName();

  if (!readMode) {
    auth::Cipher cipher0(m_principalKey);
    std::string adminKey = cipher0.b64decrypt(princData.adminKey);
    if (adminKey != m_principalKey) {
      // not admin user!
      throwException("Provided credentials does not allow admin operation.", "CredentialStore::openSession");
    }

    // first find the credentials for WRITING in the security tables
    std::unique_ptr<coral::IQuery> query(schema.newQuery());
    query->addToTableList(tname(AUTHORIZATION_TABLE, schemaVersion), "AUTHO");
    query->addToTableList(tname(CREDENTIAL_TABLE, schemaVersion), "CREDS");
    coral::AttributeList readBuff;
    readBuff.extend<std::string>("CREDS." + CONNECTION_LABEL_COL);
    readBuff.extend<std::string>("CREDS." + CONNECTION_KEY_COL);
    readBuff.extend<std::string>("CREDS." + USERNAME_COL);
    readBuff.extend<std::string>("CREDS." + PASSWORD_COL);
    readBuff.extend<std::string>("CREDS." + VERIFICATION_KEY_COL);
    coral::AttributeList whereData;
    whereData.extend<int>(P_ID_COL);
    whereData.extend<std::string>(ROLE_COL);
    whereData.extend<std::string>(SCHEMA_COL);
    whereData[P_ID_COL].data<int>() = m_principalId;
    whereData[ROLE_COL].data<std::string>() = auth::COND_ADMIN_ROLE;
    whereData[SCHEMA_COL].data<std::string>() = storeConnectionString;
    std::stringstream whereClause;
    whereClause << "AUTHO." << C_ID_COL << " = CREDS." << CONNECTION_ID_COL;
    whereClause << " AND AUTHO." << P_ID_COL << " = :" << P_ID_COL;
    whereClause << " AND AUTHO." << ROLE_COL << " = :" << ROLE_COL;
    whereClause << " AND AUTHO." << SCHEMA_COL << " = :" << SCHEMA_COL;
    query->defineOutput(readBuff);
    query->addToOutputList("CREDS." + CONNECTION_LABEL_COL);
    query->addToOutputList("CREDS." + CONNECTION_KEY_COL);
    query->addToOutputList("CREDS." + USERNAME_COL);
    query->addToOutputList("CREDS." + PASSWORD_COL);
    query->addToOutputList("CREDS." + VERIFICATION_KEY_COL);
    query->setCondition(whereClause.str(), whereData);
    coral::ICursor& cursor = query->execute();
    bool found = false;
    std::string writeUserName("");
    std::string writePassword("");
    if (cursor.next()) {
      const coral::AttributeList& row = cursor.currentRow();
      const std::string& connLabel = row["CREDS." + CONNECTION_LABEL_COL].data<std::string>();
      const std::string& encryptedConnectionKey = row["CREDS." + CONNECTION_KEY_COL].data<std::string>();
      std::string connectionKey = cipher0.b64decrypt(encryptedConnectionKey);
      auth::Cipher cipher1(connectionKey);
      const std::string& encryptedUserName = row["CREDS." + USERNAME_COL].data<std::string>();
      const std::string& encryptedPassword = row["CREDS." + PASSWORD_COL].data<std::string>();
      std::string verificationKey = cipher1.b64decrypt(row["CREDS." + VERIFICATION_KEY_COL].data<std::string>());
      if (verificationKey != connLabel) {
        throwException("Could not decrypt credentials.Provided key is invalid.", "CredentialStore::startSession");
      }
      writeUserName = cipher1.b64decrypt(encryptedUserName);
      writePassword = cipher1.b64decrypt(encryptedPassword);
      found = true;
    }
    if (!found) {
      throwException("Provided credentials are invalid for write access.", "CredentialStore::openSession");
    }
    m_session->transaction().commit();
    m_session->endUserSession();
    openSession(connTokens.second, writeUserName, writePassword, false);
  }
}

// class public methods
cond::CredentialStore::CredentialStore()
    : m_connection(),
      m_session(),
      m_authenticatedPrincipal(""),
      m_principalId(-1),
      m_principalKey(""),
      m_serviceName(""),
      m_serviceData(nullptr),
      m_key(),
      m_log() {}

cond::CredentialStore::~CredentialStore() {}

std::string cond::CredentialStore::setUpForService(const std::string& serviceName, const std::string& authPath) {
  if (serviceName.empty()) {
    throwException("Service name has not been provided.", "cond::CredentialStore::setUpConnection");
  }
  m_serviceName.clear();
  m_serviceData = nullptr;

  if (authPath.empty()) {
    throwException("The authentication Path has not been provided.", "cond::CredentialStore::setUpForService");
  }
  std::filesystem::path fullPath(authPath);
  if (!std::filesystem::exists(authPath) || !std::filesystem::is_directory(authPath)) {
    throwException("Authentication Path is invalid.", "cond::CredentialStore::setUpForService");
  }
  std::filesystem::path file(auth::DecodingKey::FILE_PATH);
  fullPath /= file;

  m_key.init(fullPath.string(), auth::COND_KEY);

  std::map<std::string, auth::ServiceCredentials>::const_iterator iK = m_key.services().find(serviceName);
  if (iK == m_key.services().end()) {
    std::string msg("");
    msg += "Service \"" + serviceName + "\" can't be open with the current key.";
    throwException(msg, "cond::CredentialStore::setUpConnection");
  }
  m_serviceName = serviceName;
  m_serviceData = &iK->second;
  m_log << "Opening Credential Store for service " << m_serviceName << " on " << m_serviceData->connectionString
        << std::endl;
  return m_serviceData->connectionString;
}

std::string cond::CredentialStore::setUpForConnectionString(const std::string& connectionString,
                                                            const std::string& authPath) {
  coral::IHandle<coral::IRelationalService> relationalService =
      coral::Context::instance().query<coral::IRelationalService>();
  if (!relationalService.isValid()) {
    coral::Context::instance().loadComponent("CORAL/Services/RelationalService");
    relationalService = coral::Context::instance().query<coral::IRelationalService>();
  }
  coral::IRelationalDomain& domain = relationalService->domainForConnection(connectionString);
  std::pair<std::string, std::string> connTokens = domain.decodeUserConnectionString(connectionString);
  std::string& serviceName = connTokens.first;
  return setUpForService(serviceName, authPath);
}

void addSequence(const std::string& schemaVersion, coral::ISchema& schema, const std::string& name) {
  // Create the entry in the table
  coral::AttributeList insertData;
  insertData.extend<std::string>(SEQUENCE_NAME_COL);
  insertData.extend<int>(SEQUENCE_VALUE_COL);
  coral::AttributeList::iterator iAttribute = insertData.begin();
  iAttribute->data<std::string>() = name;
  ++iAttribute;
  iAttribute->data<int>() = -1;
  schema.tableHandle(tname(SEQUENCE_TABLE, schemaVersion)).dataEditor().insertRow(insertData);
}

bool cond::CredentialStore::createSchema(const std::string& connectionString,
                                         const std::string& userName,
                                         const std::string& password) {
  CSScopedSession session(*this);
  session.startSuper(connectionString, userName, password);

  coral::ISchema& schema = m_session->nominalSchema();
  std::string authentication_table_name = tname(AUTHENTICATION_TABLE, m_key.version());
  if (schema.existsTable(authentication_table_name)) {
    throwException("Credential database, already exists.", "CredentialStore::create");
  }

  m_log << "Creating sequence table." << std::endl;
  std::string sequence_table_name = tname(SEQUENCE_TABLE, m_key.version());
  coral::TableDescription dseq;
  dseq.setName(sequence_table_name);
  dseq.insertColumn(SEQUENCE_NAME_COL, coral::AttributeSpecification::typeNameForType<std::string>());
  dseq.setNotNullConstraint(SEQUENCE_NAME_COL);
  dseq.insertColumn(SEQUENCE_VALUE_COL, coral::AttributeSpecification::typeNameForType<int>());
  dseq.setNotNullConstraint(SEQUENCE_VALUE_COL);
  dseq.setPrimaryKey(std::vector<std::string>(1, SEQUENCE_NAME_COL));
  schema.createTable(dseq);

  int columnSize = 2000;

  m_log << "Creating authentication table." << std::endl;
  // authentication table
  addSequence(m_key.version(), schema, authentication_table_name);
  coral::TableDescription descr0;
  descr0.setName(authentication_table_name);
  descr0.insertColumn(PRINCIPAL_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr0.insertColumn(
      PRINCIPAL_NAME_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr0.insertColumn(
      VERIFICATION_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr0.insertColumn(
      PRINCIPAL_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr0.insertColumn(ADMIN_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr0.setNotNullConstraint(PRINCIPAL_ID_COL);
  descr0.setNotNullConstraint(PRINCIPAL_NAME_COL);
  descr0.setNotNullConstraint(VERIFICATION_COL);
  descr0.setNotNullConstraint(PRINCIPAL_KEY_COL);
  descr0.setNotNullConstraint(ADMIN_KEY_COL);
  std::vector<std::string> columnsUnique;
  columnsUnique.push_back(PRINCIPAL_NAME_COL);
  descr0.setUniqueConstraint(columnsUnique);
  std::vector<std::string> columnsForIndex;
  columnsForIndex.push_back(PRINCIPAL_ID_COL);
  descr0.setPrimaryKey(columnsForIndex);
  schema.createTable(descr0);

  m_log << "Creating authorization table." << std::endl;
  std::string authorization_table_name = tname(AUTHORIZATION_TABLE, m_key.version());
  // authorization table
  addSequence(m_key.version(), schema, authorization_table_name);
  coral::TableDescription descr1;
  descr1.setName(authorization_table_name);
  descr1.insertColumn(AUTH_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr1.insertColumn(P_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr1.insertColumn(ROLE_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr1.insertColumn(SCHEMA_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr1.insertColumn(AUTH_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr1.insertColumn(C_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr1.setNotNullConstraint(AUTH_ID_COL);
  descr1.setNotNullConstraint(P_ID_COL);
  descr1.setNotNullConstraint(ROLE_COL);
  descr1.setNotNullConstraint(SCHEMA_COL);
  descr1.setNotNullConstraint(AUTH_KEY_COL);
  descr1.setNotNullConstraint(C_ID_COL);
  columnsUnique.clear();
  columnsUnique.push_back(P_ID_COL);
  columnsUnique.push_back(ROLE_COL);
  columnsUnique.push_back(SCHEMA_COL);
  descr1.setUniqueConstraint(columnsUnique);
  columnsForIndex.clear();
  columnsForIndex.push_back(AUTH_ID_COL);
  descr1.setPrimaryKey(columnsForIndex);
  schema.createTable(descr1);

  m_log << "Creating credential table." << std::endl;
  std::string credential_table_name = tname(CREDENTIAL_TABLE, m_key.version());
  // credential table
  addSequence(m_key.version(), schema, credential_table_name);
  coral::TableDescription descr2;
  descr2.setName(credential_table_name);
  descr2.insertColumn(CONNECTION_ID_COL, coral::AttributeSpecification::typeNameForType<int>());
  descr2.insertColumn(
      CONNECTION_LABEL_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr2.insertColumn(USERNAME_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr2.insertColumn(PASSWORD_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr2.insertColumn(
      VERIFICATION_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr2.insertColumn(
      CONNECTION_KEY_COL, coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false);
  descr2.setNotNullConstraint(CONNECTION_ID_COL);
  descr2.setNotNullConstraint(CONNECTION_LABEL_COL);
  descr2.setNotNullConstraint(USERNAME_COL);
  descr2.setNotNullConstraint(PASSWORD_COL);
  descr2.setNotNullConstraint(VERIFICATION_KEY_COL);
  descr2.setNotNullConstraint(CONNECTION_KEY_COL);
  columnsUnique.clear();
  columnsUnique.push_back(CONNECTION_LABEL_COL);
  descr2.setUniqueConstraint(columnsUnique);
  columnsForIndex.clear();
  columnsForIndex.push_back(CONNECTION_ID_COL);
  descr2.setPrimaryKey(columnsForIndex);
  schema.createTable(descr2);

  try {
    schema.tableHandle(authentication_table_name)
        .privilegeManager()
        .grantToUser(m_serviceData->userName, coral::ITablePrivilegeManager::Select);
    schema.tableHandle(authorization_table_name)
        .privilegeManager()
        .grantToUser(m_serviceData->userName, coral::ITablePrivilegeManager::Select);
    schema.tableHandle(credential_table_name)
        .privilegeManager()
        .grantToUser(m_serviceData->userName, coral::ITablePrivilegeManager::Select);
  } catch (const coral::Exception& e) {
    std::cout << "WARNING: Could not grant select access to user " << m_serviceData->userName << ": [" << e.what()
              << "]" << std::endl;
  }
  m_log << "Granting ADMIN access permission." << std::endl;
  auth::KeyGenerator gen;
  m_principalKey = gen.make(auth::COND_DB_KEY_SIZE);
  auto princData = updatePrincipalData(
      m_key.version(), schema, m_key.principalKey(), m_key.principalName(), m_principalKey, true, m_log);
  std::string credentialAccessLabel = schemaLabel(m_serviceName, userName);
  auto connParams = updateConnectionData(
      m_key.version(), schema, m_principalKey, credentialAccessLabel, userName, password, true, m_log);
  bool ret = setPermissionData(m_key.version(),
                               schema,
                               princData.first,
                               m_principalKey,
                               auth::COND_ADMIN_ROLE,
                               connectionString,
                               connParams.first,
                               connParams.second,
                               m_log);
  session.close();
  return ret;
}

bool cond::CredentialStore::drop(const std::string& connectionString,
                                 const std::string& userName,
                                 const std::string& password) {
  CSScopedSession session(*this);
  session.startSuper(connectionString, userName, password);

  m_log << "Dropping AUTHORIZATION, CREDENTIAL, AUTHENTICATION and SEQUENCE tables." << std::endl;
  coral::ISchema& schema = m_session->nominalSchema();
  schema.dropIfExistsTable(tname(AUTHORIZATION_TABLE, m_key.version()));
  schema.dropIfExistsTable(tname(CREDENTIAL_TABLE, m_key.version()));
  schema.dropIfExistsTable(tname(AUTHENTICATION_TABLE, m_key.version()));
  schema.dropIfExistsTable(tname(SEQUENCE_TABLE, m_key.version()));
  session.close();
  return true;
}

bool cond::CredentialStore::resetAdmin(const std::string& userName, const std::string& password) {
  if (!m_serviceData) {
    throwException("The credential store has not been initialized.", "cond::CredentialStore::installAdmin");
  }
  const std::string& connectionString = m_serviceData->connectionString;

  CSScopedSession session(*this);
  session.startSuper(connectionString, userName, password);

  coral::ISchema& schema = m_session->nominalSchema();
  const std::string& principalName = m_key.principalName();
  const std::string& authenticationKey = m_key.principalKey();
  PrincipalData princData;
  if (!selectPrincipal(m_key.version(), schema, principalName, princData)) {
    std::string msg("User \"");
    msg += principalName + "\" has not been found.";
    throwException(msg, "CredentialStore::resetAdmin");
  }
  auth::Cipher cipher0(authenticationKey);
  m_principalKey = cipher0.b64decrypt(princData.principalKey);

  auto p = updatePrincipalData(m_key.version(), schema, authenticationKey, principalName, m_principalKey, false, m_log);
  std::string credentialAccessLabel = schemaLabel(m_serviceName, userName);
  auto connParams = updateConnectionData(
      m_key.version(), schema, m_principalKey, credentialAccessLabel, userName, password, true, m_log);
  bool ret = setPermissionData(m_key.version(),
                               schema,
                               p.first,
                               m_principalKey,
                               auth::COND_ADMIN_ROLE,
                               connectionString,
                               connParams.first,
                               connParams.second,
                               m_log);
  session.close();
  return ret;
}

bool cond::CredentialStore::updatePrincipal(const std::string& principalName,
                                            const std::string& authenticationKey,
                                            bool setAdmin) {
  CSScopedSession session(*this);
  session.start(false);
  coral::ISchema& schema = m_session->nominalSchema();
  auto princData =
      updatePrincipalData(m_key.version(), schema, authenticationKey, principalName, m_principalKey, false, m_log);
  bool ret = false;
  if (setAdmin) {
    int princId = princData.first;
    std::string princKey = m_principalKey;
    std::string connString = m_serviceData->connectionString;
    std::vector<Permission> permissions;
    if (!selectPermissions(m_key.principalName(), auth::COND_ADMIN_ROLE, connString, permissions)) {
      throwException("The current operating user is not admin user on the underlying Credential Store.",
                     "CredentialStore::updatePrincipal");
    }
    std::string connLabel = permissions.front().connectionLabel;
    CredentialData credsData;
    if (!selectConnection(m_key.version(), schema, connLabel, credsData)) {
      throwException("Credential Store connection has not been defined.", "CredentialStore::updatePrincipal");
    }
    auth::Cipher adminCipher(m_principalKey);
    ret = setPermissionData(m_key.version(),
                            schema,
                            princId,
                            princKey,
                            auth::COND_ADMIN_ROLE,
                            connString,
                            credsData.id,
                            adminCipher.b64decrypt(credsData.connectionKey),
                            m_log);
  }
  session.close();
  return ret;
}

bool cond::CredentialStore::setPermission(const std::string& principal,
                                          const std::string& role,
                                          const std::string& connectionString,
                                          const std::string& connectionLabel) {
  CSScopedSession session(*this);
  session.start(false);

  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal(m_key.version(), schema, principal, princData);

  if (!found) {
    std::string msg = "Principal \"" + principal + "\" does not exist in the database.";
    throwException(msg, "CredentialStore::setPermission");
  }

  m_log << "Principal " << principal << " id: " << princData.id << std::endl;
  CredentialData credsData;
  found = selectConnection(m_key.version(), schema, connectionLabel, credsData);

  if (!found) {
    std::string msg = "Connection named \"" + connectionLabel + "\" does not exist in the database.";
    throwException(msg, "CredentialStore::setPermission");
  }

  auth::Cipher cipher(m_principalKey);
  bool ret = setPermissionData(m_key.version(),
                               schema,
                               princData.id,
                               cipher.b64decrypt(princData.adminKey),
                               role,
                               connectionString,
                               credsData.id,
                               cipher.b64decrypt(credsData.connectionKey),
                               m_log);
  session.close();
  return ret;
}

size_t cond::CredentialStore::unsetPermission(const std::string& principal,
                                              const std::string& role,
                                              const std::string& connectionString) {
  if (!role.empty() && cond::auth::ROLES.find(role) == cond::auth::ROLES.end()) {
    throwException(std::string("Role ") + role + " does not exists.", "CredentialStore::unsetPermission");
  }
  CSScopedSession session(*this);
  session.start(false);
  coral::ISchema& schema = m_session->nominalSchema();

  coral::AttributeList deleteData;
  deleteData.extend<std::string>(SCHEMA_COL);
  std::stringstream whereClause;
  m_log << "Removing permissions to access resource " << connectionString;
  if (!role.empty()) {
    deleteData.extend<std::string>(ROLE_COL);
    m_log << " with role " << role;
  }
  int princId = -1;
  if (!principal.empty()) {
    PrincipalData princData;
    bool found = selectPrincipal(m_key.version(), schema, principal, princData);

    if (!found) {
      std::string msg = "Principal \"" + principal + "\" does not exist in the database.";
      throwException(msg, "CredentialStore::unsetPermission");
    }
    deleteData.extend<int>(P_ID_COL);
    princId = princData.id;
    m_log << " by principal " << principal << " (id: " << princData.id << ")";
  }

  size_t n_e = getAuthorizationEntries(m_key.version(), schema, princId, role, connectionString);
  m_log << ": " << n_e << " authorization entries." << std::endl;
  if (n_e) {
    deleteData[SCHEMA_COL].data<std::string>() = connectionString;
    whereClause << SCHEMA_COL << " = :" << SCHEMA_COL;
    if (!role.empty()) {
      deleteData[ROLE_COL].data<std::string>() = role;
      whereClause << " AND " << ROLE_COL << " = :" << ROLE_COL;
    }
    if (!principal.empty()) {
      deleteData[P_ID_COL].data<int>() = princId;
      whereClause << " AND " << P_ID_COL + " = :" + P_ID_COL;
    }
    coral::ITableDataEditor& editor = schema.tableHandle(tname(AUTHORIZATION_TABLE, m_key.version())).dataEditor();
    editor.deleteRows(whereClause.str(), deleteData);
  }
  session.close();
  return n_e;
}

bool cond::CredentialStore::updateConnection(const std::string& connectionLabel,
                                             const std::string& userName,
                                             const std::string& password) {
  CSScopedSession session(*this);
  session.start(false);

  m_session->transaction().start();
  coral::ISchema& schema = m_session->nominalSchema();
  std::string connLabel = to_lower(connectionLabel);
  updateConnectionData(m_key.version(), schema, m_principalKey, connLabel, userName, password, true, m_log);

  session.close();
  return true;
}

bool cond::CredentialStore::removePrincipal(const std::string& principal) {
  CSScopedSession session(*this);
  session.start(false);
  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal(m_key.version(), schema, principal, princData);

  if (!found) {
    std::string msg = "Principal \"" + principal + "\" does not exist in the database.";
    throwException(msg, "CredentialStore::removePrincipal");
  }

  m_log << "Removing principal " << principal << " (id: " << princData.id << ")" << std::endl;

  coral::ITableDataEditor& editor0 = schema.tableHandle(tname(AUTHORIZATION_TABLE, m_key.version())).dataEditor();

  coral::AttributeList deleteData0;
  deleteData0.extend<int>(P_ID_COL);
  deleteData0[P_ID_COL].data<int>() = princData.id;
  std::string whereClause0 = P_ID_COL + " = :" + P_ID_COL;
  editor0.deleteRows(whereClause0, deleteData0);

  coral::ITableDataEditor& editor1 = schema.tableHandle(tname(AUTHENTICATION_TABLE, m_key.version())).dataEditor();

  coral::AttributeList deleteData1;
  deleteData1.extend<int>(PRINCIPAL_ID_COL);
  deleteData1[PRINCIPAL_ID_COL].data<int>() = princData.id;
  std::string whereClause1 = PRINCIPAL_ID_COL + " = :" + PRINCIPAL_ID_COL;
  editor1.deleteRows(whereClause1, deleteData1);

  session.close();

  return true;
}

bool cond::CredentialStore::removeConnection(const std::string& connectionLabel) {
  CSScopedSession session(*this);
  session.start(false);
  coral::ISchema& schema = m_session->nominalSchema();

  CredentialData credsData;
  bool found = selectConnection(m_key.version(), schema, connectionLabel, credsData);

  if (!found) {
    std::string msg = "Connection named \"" + connectionLabel + "\" does not exist in the database.";
    throwException(msg, "CredentialStore::removeConnection");
  }

  m_log << "Removing connection " << connectionLabel << std::endl;
  coral::ITableDataEditor& editor0 = schema.tableHandle(tname(AUTHORIZATION_TABLE, m_key.version())).dataEditor();

  coral::AttributeList deleteData0;
  deleteData0.extend<int>(C_ID_COL);
  deleteData0[C_ID_COL].data<int>() = credsData.id;
  std::string whereClause0 = C_ID_COL + " = :" + C_ID_COL;
  editor0.deleteRows(whereClause0, deleteData0);

  coral::ITableDataEditor& editor1 = schema.tableHandle(tname(CREDENTIAL_TABLE, m_key.version())).dataEditor();

  coral::AttributeList deleteData1;
  deleteData1.extend<int>(CONNECTION_ID_COL);
  deleteData1[CONNECTION_ID_COL].data<int>() = credsData.id;
  std::string whereClause1 = CONNECTION_ID_COL + " = :" + CONNECTION_ID_COL;
  editor1.deleteRows(whereClause1, deleteData1);

  session.close();

  return true;
}

bool cond::CredentialStore::selectForUser(coral_bridge::AuthenticationCredentialSet& destinationData) {
  CSScopedSession session(*this);
  session.start(true);
  coral::ISchema& schema = m_session->nominalSchema();

  auth::Cipher cipher(m_principalKey);

  std::unique_ptr<coral::IQuery> query(schema.newQuery());
  query->addToTableList(tname(AUTHORIZATION_TABLE, m_key.version()), "AUTHO");
  query->addToTableList(tname(CREDENTIAL_TABLE, m_key.version()), "CREDS");
  coral::AttributeList readBuff;
  readBuff.extend<std::string>("AUTHO." + ROLE_COL);
  readBuff.extend<std::string>("AUTHO." + SCHEMA_COL);
  readBuff.extend<std::string>("AUTHO." + AUTH_KEY_COL);
  readBuff.extend<std::string>("CREDS." + CONNECTION_LABEL_COL);
  readBuff.extend<std::string>("CREDS." + USERNAME_COL);
  readBuff.extend<std::string>("CREDS." + PASSWORD_COL);
  readBuff.extend<std::string>("CREDS." + VERIFICATION_KEY_COL);
  coral::AttributeList whereData;
  whereData.extend<int>(P_ID_COL);
  whereData[P_ID_COL].data<int>() = m_principalId;
  std::stringstream whereClause;
  whereClause << "AUTHO." << C_ID_COL << "="
              << "CREDS." << CONNECTION_ID_COL;
  whereClause << " AND "
              << "AUTHO." << P_ID_COL << " = :" << P_ID_COL;
  query->defineOutput(readBuff);
  query->addToOutputList("AUTHO." + ROLE_COL);
  query->addToOutputList("AUTHO." + SCHEMA_COL);
  query->addToOutputList("AUTHO." + AUTH_KEY_COL);
  query->addToOutputList("CREDS." + CONNECTION_LABEL_COL);
  query->addToOutputList("CREDS." + USERNAME_COL);
  query->addToOutputList("CREDS." + PASSWORD_COL);
  query->addToOutputList("CREDS." + VERIFICATION_KEY_COL);
  query->setCondition(whereClause.str(), whereData);
  coral::ICursor& cursor = query->execute();
  while (cursor.next()) {
    const coral::AttributeList& row = cursor.currentRow();
    const std::string& role = row["AUTHO." + ROLE_COL].data<std::string>();
    const std::string& connectionString = row["AUTHO." + SCHEMA_COL].data<std::string>();
    const std::string& encryptedAuthKey = row["AUTHO." + AUTH_KEY_COL].data<std::string>();
    const std::string& connectionLabel = row["CREDS." + CONNECTION_LABEL_COL].data<std::string>();
    const std::string& encryptedUserName = row["CREDS." + USERNAME_COL].data<std::string>();
    const std::string& encryptedPassword = row["CREDS." + PASSWORD_COL].data<std::string>();
    std::string authKey = cipher.b64decrypt(encryptedAuthKey);
    auth::Cipher connCipher(authKey);
    std::string verificationString = connCipher.b64decrypt(row["CREDS." + VERIFICATION_KEY_COL].data<std::string>());
    if (verificationString == connectionLabel) {
      destinationData.registerCredentials(to_lower(connectionString),
                                          role,
                                          connCipher.b64decrypt(encryptedUserName),
                                          connCipher.b64decrypt(encryptedPassword));
    }
  }
  session.close();
  return true;
}

std::pair<std::string, std::string> cond::CredentialStore::getUserCredentials(const std::string& connectionString,
                                                                              const std::string& role) {
  CSScopedSession session(*this);
  session.start(true);
  coral::ISchema& schema = m_session->nominalSchema();

  auth::Cipher cipher(m_principalKey);

  std::unique_ptr<coral::IQuery> query(schema.newQuery());
  query->addToTableList(tname(AUTHORIZATION_TABLE, m_key.version()), "AUTHO");
  query->addToTableList(tname(CREDENTIAL_TABLE, m_key.version()), "CREDS");
  coral::AttributeList readBuff;
  readBuff.extend<std::string>("AUTHO." + AUTH_KEY_COL);
  readBuff.extend<std::string>("CREDS." + CONNECTION_LABEL_COL);
  readBuff.extend<std::string>("CREDS." + USERNAME_COL);
  readBuff.extend<std::string>("CREDS." + PASSWORD_COL);
  readBuff.extend<std::string>("CREDS." + VERIFICATION_KEY_COL);
  coral::AttributeList whereData;
  whereData.extend<int>(P_ID_COL);
  whereData.extend<std::string>(SCHEMA_COL);
  whereData.extend<std::string>(ROLE_COL);
  whereData[P_ID_COL].data<int>() = m_principalId;
  whereData[SCHEMA_COL].data<std::string>() = to_lower(connectionString);
  whereData[ROLE_COL].data<std::string>() = role;
  std::stringstream whereClause;
  whereClause << "AUTHO." << C_ID_COL << "="
              << "CREDS." << CONNECTION_ID_COL;
  whereClause << " AND "
              << "AUTHO." << P_ID_COL << " = :" << P_ID_COL;
  whereClause << " AND "
              << "AUTHO." << SCHEMA_COL << " = :" << SCHEMA_COL;
  whereClause << " AND "
              << "AUTHO." << ROLE_COL << " = :" << ROLE_COL;
  query->defineOutput(readBuff);
  query->addToOutputList("AUTHO." + AUTH_KEY_COL);
  query->addToOutputList("CREDS." + CONNECTION_LABEL_COL);
  query->addToOutputList("CREDS." + USERNAME_COL);
  query->addToOutputList("CREDS." + PASSWORD_COL);
  query->addToOutputList("CREDS." + VERIFICATION_KEY_COL);
  query->setCondition(whereClause.str(), whereData);
  coral::ICursor& cursor = query->execute();
  auto ret = std::make_pair(std::string(""), std::string(""));
  if (cursor.next()) {
    const coral::AttributeList& row = cursor.currentRow();
    const std::string& encryptedAuthKey = row["AUTHO." + AUTH_KEY_COL].data<std::string>();
    const std::string& connectionLabel = row["CREDS." + CONNECTION_LABEL_COL].data<std::string>();
    const std::string& encryptedUserName = row["CREDS." + USERNAME_COL].data<std::string>();
    const std::string& encryptedPassword = row["CREDS." + PASSWORD_COL].data<std::string>();
    std::string authKey = cipher.b64decrypt(encryptedAuthKey);
    auth::Cipher connCipher(authKey);
    std::string verificationString = connCipher.b64decrypt(row["CREDS." + VERIFICATION_KEY_COL].data<std::string>());
    if (verificationString == connectionLabel) {
      ret.first = connCipher.b64decrypt(encryptedUserName);
      ret.second = connCipher.b64decrypt(encryptedPassword);
    }
  }
  session.close();
  return ret;
}

bool cond::CredentialStore::importForPrincipal(const std::string& principal,
                                               const coral_bridge::AuthenticationCredentialSet& dataSource,
                                               bool forceUpdateConnection) {
  CSScopedSession session(*this);
  session.start(false);
  coral::ISchema& schema = m_session->nominalSchema();

  PrincipalData princData;
  bool found = selectPrincipal(m_key.version(), schema, principal, princData);

  if (!found) {
    std::string msg = "Principal \"" + principal + "\" does not exist in the database.";
    throwException(msg, "CredentialStore::importForPrincipal");
  }

  bool imported = false;
  auth::Cipher cipher(m_principalKey);
  std::string princKey = cipher.b64decrypt(princData.adminKey);

  const std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>& creds = dataSource.data();
  for (std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>::const_iterator iConn =
           creds.begin();
       iConn != creds.end();
       ++iConn) {
    const std::string& connectionString = iConn->first.first;
    coral::URIParser parser;
    parser.setURI(connectionString);
    std::string serviceName = parser.hostName();
    const std::string& role = iConn->first.second;
    std::string userName = iConn->second->valueForItem(coral::IAuthenticationCredentials::userItem());
    std::string password = iConn->second->valueForItem(coral::IAuthenticationCredentials::passwordItem());
    // first import the connections
    std::pair<int, std::string> conn = updateConnectionData(m_key.version(),
                                                            schema,
                                                            m_principalKey,
                                                            schemaLabel(serviceName, userName),
                                                            userName,
                                                            password,
                                                            forceUpdateConnection,
                                                            m_log);
    auth::Cipher cipher(m_principalKey);
    // than set the permission for the specific role
    setPermissionData(
        m_key.version(), schema, princData.id, princKey, role, connectionString, conn.first, conn.second, m_log);
    imported = true;
  }
  session.close();
  return imported;
}

bool cond::CredentialStore::listPrincipals(std::vector<std::string>& destination) {
  CSScopedSession session(*this);
  session.start(true);
  coral::ISchema& schema = m_session->nominalSchema();

  std::unique_ptr<coral::IQuery> query(schema.tableHandle(tname(AUTHENTICATION_TABLE, m_key.version())).newQuery());
  coral::AttributeList readBuff;
  readBuff.extend<std::string>(PRINCIPAL_NAME_COL);
  query->defineOutput(readBuff);
  query->addToOutputList(PRINCIPAL_NAME_COL);
  coral::ICursor& cursor = query->execute();
  bool found = false;
  while (cursor.next()) {
    found = true;
    const coral::AttributeList& row = cursor.currentRow();
    destination.push_back(row[PRINCIPAL_NAME_COL].data<std::string>());
  }
  session.close();
  return found;
}

bool cond::CredentialStore::listConnections(std::map<std::string, std::pair<std::string, std::string> >& destination) {
  CSScopedSession session(*this);
  session.start(true);
  coral::ISchema& schema = m_session->nominalSchema();

  std::unique_ptr<coral::IQuery> query(schema.tableHandle(tname(CREDENTIAL_TABLE, m_key.version())).newQuery());
  coral::AttributeList readBuff;
  readBuff.extend<std::string>(CONNECTION_LABEL_COL);
  readBuff.extend<std::string>(USERNAME_COL);
  readBuff.extend<std::string>(PASSWORD_COL);
  readBuff.extend<std::string>(VERIFICATION_KEY_COL);
  readBuff.extend<std::string>(CONNECTION_KEY_COL);
  query->defineOutput(readBuff);
  query->addToOutputList(CONNECTION_LABEL_COL);
  query->addToOutputList(USERNAME_COL);
  query->addToOutputList(PASSWORD_COL);
  query->addToOutputList(VERIFICATION_KEY_COL);
  query->addToOutputList(CONNECTION_KEY_COL);
  coral::ICursor& cursor = query->execute();
  bool found = false;
  auth::Cipher cipher0(m_principalKey);
  while (cursor.next()) {
    std::string userName("");
    std::string password("");
    const coral::AttributeList& row = cursor.currentRow();
    const std::string& connLabel = row[CONNECTION_LABEL_COL].data<std::string>();
    const std::string& encryptedKey = row[CONNECTION_KEY_COL].data<std::string>();
    const std::string& encryptedVerif = row[VERIFICATION_KEY_COL].data<std::string>();
    std::string connKey = cipher0.b64decrypt(encryptedKey);
    auth::Cipher cipher1(connKey);
    std::string verif = cipher1.b64decrypt(encryptedVerif);
    if (verif == connLabel) {
      const std::string& encryptedUserName = row[USERNAME_COL].data<std::string>();
      const std::string& encryptedPassword = row[PASSWORD_COL].data<std::string>();
      userName = cipher1.b64decrypt(encryptedUserName);
      password = cipher1.b64decrypt(encryptedPassword);
    }
    destination.insert(std::make_pair(connLabel, std::make_pair(userName, password)));
    found = true;
  }
  session.close();
  return found;
}

bool cond::CredentialStore::selectPermissions(const std::string& principalName,
                                              const std::string& role,
                                              const std::string& connectionString,
                                              std::vector<Permission>& destination) {
  CSScopedSession session(*this);
  session.start(true);
  coral::ISchema& schema = m_session->nominalSchema();
  std::unique_ptr<coral::IQuery> query(schema.newQuery());
  query->addToTableList(tname(AUTHENTICATION_TABLE, m_key.version()), "AUTHE");
  query->addToTableList(tname(AUTHORIZATION_TABLE, m_key.version()), "AUTHO");
  query->addToTableList(tname(CREDENTIAL_TABLE, m_key.version()), "CREDS");
  coral::AttributeList readBuff;
  readBuff.extend<std::string>("AUTHE." + PRINCIPAL_NAME_COL);
  readBuff.extend<std::string>("AUTHO." + ROLE_COL);
  readBuff.extend<std::string>("AUTHO." + SCHEMA_COL);
  readBuff.extend<std::string>("CREDS." + CONNECTION_LABEL_COL);
  coral::AttributeList whereData;
  std::stringstream whereClause;
  whereClause << "AUTHE." << PRINCIPAL_ID_COL << "= AUTHO." << P_ID_COL;
  whereClause << " AND AUTHO." << C_ID_COL << "="
              << "CREDS." << CONNECTION_ID_COL;
  if (!principalName.empty()) {
    whereData.extend<std::string>(PRINCIPAL_NAME_COL);
    whereData[PRINCIPAL_NAME_COL].data<std::string>() = principalName;
    whereClause << " AND AUTHE." << PRINCIPAL_NAME_COL << " = :" << PRINCIPAL_NAME_COL;
  }
  if (!role.empty()) {
    whereData.extend<std::string>(ROLE_COL);
    whereData[ROLE_COL].data<std::string>() = role;
    whereClause << " AND AUTHO." << ROLE_COL << " = :" << ROLE_COL;
  }
  if (!connectionString.empty()) {
    whereData.extend<std::string>(SCHEMA_COL);
    whereData[SCHEMA_COL].data<std::string>() = to_lower(connectionString);
    whereClause << " AND AUTHO." << SCHEMA_COL << " = :" << SCHEMA_COL;
  }

  query->defineOutput(readBuff);
  query->addToOutputList("AUTHE." + PRINCIPAL_NAME_COL);
  query->addToOutputList("AUTHO." + ROLE_COL);
  query->addToOutputList("AUTHO." + SCHEMA_COL);
  query->addToOutputList("CREDS." + CONNECTION_LABEL_COL);
  query->setCondition(whereClause.str(), whereData);
  query->addToOrderList("AUTHO." + SCHEMA_COL);
  query->addToOrderList("AUTHE." + PRINCIPAL_NAME_COL);
  query->addToOrderList("AUTHO." + ROLE_COL);
  coral::ICursor& cursor = query->execute();
  bool found = false;
  while (cursor.next()) {
    const coral::AttributeList& row = cursor.currentRow();
    destination.resize(destination.size() + 1);
    Permission& perm = destination.back();
    perm.principalName = row["AUTHE." + PRINCIPAL_NAME_COL].data<std::string>();
    perm.role = row["AUTHO." + ROLE_COL].data<std::string>();
    perm.connectionString = row["AUTHO." + SCHEMA_COL].data<std::string>();
    perm.connectionLabel = row["CREDS." + CONNECTION_LABEL_COL].data<std::string>();
    found = true;
  }
  session.close();
  return found;
}

bool cond::CredentialStore::exportAll(coral_bridge::AuthenticationCredentialSet& data) {
  CSScopedSession session(*this);
  session.start(true);
  coral::ISchema& schema = m_session->nominalSchema();
  std::unique_ptr<coral::IQuery> query(schema.newQuery());
  query->addToTableList(tname(AUTHORIZATION_TABLE, m_key.version()), "AUTHO");
  query->addToTableList(tname(CREDENTIAL_TABLE, m_key.version()), "CREDS");
  coral::AttributeList readBuff;
  readBuff.extend<std::string>("AUTHO." + ROLE_COL);
  readBuff.extend<std::string>("AUTHO." + SCHEMA_COL);
  readBuff.extend<std::string>("CREDS." + CONNECTION_LABEL_COL);
  readBuff.extend<std::string>("CREDS." + VERIFICATION_KEY_COL);
  readBuff.extend<std::string>("CREDS." + CONNECTION_KEY_COL);
  readBuff.extend<std::string>("CREDS." + USERNAME_COL);
  readBuff.extend<std::string>("CREDS." + PASSWORD_COL);
  coral::AttributeList whereData;
  std::stringstream whereClause;
  whereClause << "AUTHO." << C_ID_COL << "="
              << "CREDS." << CONNECTION_ID_COL;

  query->defineOutput(readBuff);
  query->addToOutputList("AUTHO." + ROLE_COL);
  query->addToOutputList("AUTHO." + SCHEMA_COL);
  query->addToOutputList("CREDS." + CONNECTION_LABEL_COL);
  query->addToOutputList("CREDS." + VERIFICATION_KEY_COL);
  query->addToOutputList("CREDS." + CONNECTION_KEY_COL);
  query->addToOutputList("CREDS." + USERNAME_COL);
  query->addToOutputList("CREDS." + PASSWORD_COL);
  query->setCondition(whereClause.str(), whereData);
  coral::ICursor& cursor = query->execute();
  bool found = false;
  auth::Cipher cipher0(m_principalKey);
  while (cursor.next()) {
    const coral::AttributeList& row = cursor.currentRow();
    const std::string& role = row["AUTHO." + ROLE_COL].data<std::string>();
    const std::string& connectionString = row["AUTHO." + SCHEMA_COL].data<std::string>();
    const std::string& connectionLabel = row["CREDS." + CONNECTION_LABEL_COL].data<std::string>();
    const std::string& encryptedVerifKey = row["CREDS." + VERIFICATION_KEY_COL].data<std::string>();
    const std::string& encryptedConnection = row["CREDS." + CONNECTION_KEY_COL].data<std::string>();
    std::string userName("");
    std::string password("");
    std::string connectionKey = cipher0.b64decrypt(encryptedConnection);
    auth::Cipher cipher1(connectionKey);
    std::string verifKey = cipher1.b64decrypt(encryptedVerifKey);
    if (verifKey == connectionLabel) {
      const std::string& encryptedUserName = row["CREDS." + USERNAME_COL].data<std::string>();
      const std::string& encryptedPassword = row["CREDS." + PASSWORD_COL].data<std::string>();
      userName = cipher1.b64decrypt(encryptedUserName);
      password = cipher1.b64decrypt(encryptedPassword);
    }
    data.registerCredentials(to_lower(connectionString), role, userName, password);
    found = true;
  }
  session.close();
  return found;
}

const std::string& cond::CredentialStore::serviceName() { return m_serviceName; }

const std::string& cond::CredentialStore::keyPrincipalName() { return m_authenticatedPrincipal; }

std::string cond::CredentialStore::log() { return m_log.str(); }
