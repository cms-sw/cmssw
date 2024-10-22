#ifndef CondCore_CondDB_CredentialStore_h
#define CondCore_CondDB_CredentialStore_h

#include "CondCore/CondDB/interface/DecodingKey.h"
//
#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <algorithm>
//
#include "CoralBase/MessageStream.h"

namespace coral {

  class AuthenticationCredentials;
  class IAuthenticationCredentials;
  class ISession;
  class IConnection;

}  // namespace coral

inline std::string to_lower(const std::string& s) {
  std::string str(s);
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
  return str;
}

namespace coral_bridge {

  class AuthenticationCredentialSet {
  public:
    /// Constructor
    AuthenticationCredentialSet();

    /// Destructor
    virtual ~AuthenticationCredentialSet();

    void registerItem(const std::string& connectionString, const std::string& itemName, const std::string& itemValue);

    void registerItem(const std::string& connectionString,
                      const std::string& role,
                      const std::string& itemName,
                      const std::string& itemValue);

    /**
     * Adds a credential item to the default role.
     */
    void registerCredentials(const std::string& connectionString,
                             const std::string& userName,
                             const std::string& password);

    /**
     * Adds a credential item to the specified role.
     */
    void registerCredentials(const std::string& connectionString,
                             const std::string& role,
                             const std::string& userName,
                             const std::string& password);

    void import(const AuthenticationCredentialSet& data);

    const coral::IAuthenticationCredentials* get(const std::string& connectionString) const;

    const coral::IAuthenticationCredentials* get(const std::string& connectionString, const std::string& role) const;

    const std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*>& data() const;

    void reset();

  private:
    /// credentials for the specific roles
    std::map<std::pair<std::string, std::string>, coral::AuthenticationCredentials*> m_data;
  };

}  // namespace coral_bridge

namespace cond {

  class Cipher;

  std::string schemaLabel(const std::string& serviceName, const std::string& userName);

  //
  class CredentialStore {
  public:
    // default service is pointed in case the specific one has not been found in the key list
    static const std::string DEFAULT_DATA_SOURCE;

  public:
    /// Standard Constructor
    CredentialStore();

    /// Standard Destructor
    virtual ~CredentialStore();

  public:
    /// Sets the initialization parameters
    std::string setUpForService(const std::string& serviceName, const std::string& authPath);

    std::string setUpForConnectionString(const std::string& connectionString, const std::string& authPath);

    bool createSchema(const std::string& connectionString, const std::string& userName, const std::string& password);

    bool drop(const std::string& connectionString, const std::string& userName, const std::string& password);

    bool resetAdmin(const std::string& userName, const std::string& password);

    bool updatePrincipal(const std::string& principal, const std::string& principalKey, bool setAdmin = false);

    bool setPermission(const std::string& principal,
                       const std::string& role,
                       const std::string& connectionString,
                       const std::string& connectionLabel);

    size_t unsetPermission(const std::string& principal, const std::string& role, const std::string& connectionString);

    bool updateConnection(const std::string& connectionLabel, const std::string& userName, const std::string& password);

    bool removePrincipal(const std::string& principal);

    bool removeConnection(const std::string& connectionLabel);

    bool selectForUser(coral_bridge::AuthenticationCredentialSet& destinationData);

    /// import data
    bool importForPrincipal(const std::string& principal,
                            const coral_bridge::AuthenticationCredentialSet& data,
                            bool forceUpdateConnection = false);

    bool listPrincipals(std::vector<std::string>& destination);

    bool listConnections(std::map<std::string, std::pair<std::string, std::string> >& destination);

    struct Permission {
      std::string principalName;
      std::string role;
      std::string connectionString;
      std::string connectionLabel;
    };
    bool selectPermissions(const std::string& principalName,
                           const std::string& role,
                           const std::string& connectionString,
                           std::vector<Permission>& destination);

    std::pair<std::string, std::string> getUserCredentials(const std::string& connectionString,
                                                           const std::string& role);

    bool exportAll(coral_bridge::AuthenticationCredentialSet& data);

    const std::string& serviceName();

    const std::string& keyPrincipalName();

    std::string log();

  private:
    friend class CSScopedSession;

    std::pair<std::string, std::string> openConnection(const std::string& connectionString);
    void openSession(const std::string& schemaName,
                     const std::string& userName,
                     const std::string& password,
                     bool readMode);
    void startSuperSession(const std::string& connectionString,
                           const std::string& userName,
                           const std::string& password);
    void startSession(bool readMode);

    void openSession(bool readOnly = true);

    void closeSession(bool commit = true);

  private:
    std::shared_ptr<coral::IConnection> m_connection;
    std::shared_ptr<coral::ISession> m_session;

    std::string m_authenticatedPrincipal;
    int m_principalId;
    // the key used to encrypt the db credentials accessibles by the owner of the authenticated key.
    std::string m_principalKey;

    std::string m_serviceName;
    const auth::ServiceCredentials* m_serviceData;

    auth::DecodingKey m_key;

    std::stringstream m_log;
  };

}  // namespace cond

#endif
