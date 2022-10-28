#ifndef COND_XMLAUTHENTITACTIONSERVICE_H
#define COND_XMLAUTHENTITACTIONSERVICE_H

#include "RelationalAccess/IAuthenticationService.h"
#include "CoralKernel/Service.h"
#include "CoralKernel/Property.h"
#include <map>
#include <set>
#include <string>
#include <boost/thread.hpp>

namespace coral {

  class AuthenticationCredentials;
  //class IAuthenticationCredentials;
}  // namespace coral

namespace cond {

  namespace XMLAuthenticationService {

    /**
     * @class DataSourceEntry
     *
     * A simple class holding the roles and the credentials corresponding to a database service
     */
    class DataSourceEntry {
    public:
      /// Constructor
      DataSourceEntry(const std::string& serviceName, const std::string& connectionName);
      DataSourceEntry() = delete;
      DataSourceEntry(const DataSourceEntry&) = delete;
      DataSourceEntry& operator=(const DataSourceEntry&) = delete;

      /// Destructor
      ~DataSourceEntry();

      /**
       * Adds a credential item to the default role.
       */
      void appendCredentialItem(const std::string& item, const std::string& value);

      /**
       * Adds a credential item to the default role.
       */
      void appendCredentialItemForRole(const std::string& item, const std::string& value, const std::string& role);

      /**
       * Returns a reference to the credentials object for the default role.
       */
      const coral::IAuthenticationCredentials& credentials() const;

      /**
       * Returns a reference to the credentials object for a given role.
       * If the role is not known to the service an UnknownRoleException is thrown.
       */
      const coral::IAuthenticationCredentials& credentials(const std::string& role) const;

    private:
      /// The service name
      std::string m_serviceName;

      /// The connection name
      std::string m_connectionName;

      /// The input file with the data
      coral::AuthenticationCredentials* m_default;

      /// The structure with the authentication data for the various roles
      std::map<std::string, coral::AuthenticationCredentials*> m_data;
    };

    /**
     * @class AuthenticationService AuthenticationService.h
     *
     * A simple implementation of the IAuthenticationService interface based on reading an XMl file
     */
    class XMLAuthenticationService : public coral::Service, virtual public coral::IAuthenticationService {
    public:
      /// Standard Constructor
      explicit XMLAuthenticationService(const std::string& name);

      /// Standard Destructor
      ~XMLAuthenticationService() override;

    public:
      /**
       * Returns a reference to the credentials object for a given connection string.
       * If the connection string is not known to the service an UnknownConnectionException is thrown.
       */
      const coral::IAuthenticationCredentials& credentials(const std::string& connectionString) const override;

      /**
       * Returns a reference to the credentials object for a given connection string.
       * If the connection string is not known to the service an UnknownConnectionException is thrown.
       * If the role is not known to the service an UnknownRoleException is thrown.
       */
      const coral::IAuthenticationCredentials& credentials(const std::string& connectionString,
                                                           const std::string& role) const override;

    public:
      /// Sets the input file name
      void setAuthenticationPath(const std::string& inputPath);

    private:
      /// Service framework related initialization
      bool initialize();

      /// Reset parsed data
      void reset();

      /// Parses an xml file
      bool processFile(const std::string& inputFileName);

      /// Verifies the existence of the authentication files
      std::set<std::string> verifyFileName();

      /// Flag indicating whether the service has been initialized
      bool m_isInitialized;

      /// The input file with the data
      std::string m_inputFileName;

      /// The structure with the authentication data
      std::map<std::string, DataSourceEntry*> m_data;

      /// the mutex lock
      mutable boost::mutex m_mutexLock;

      coral::Property::CallbackID m_callbackID;
    };

  }  // namespace XMLAuthenticationService

}  // namespace cond

#endif
