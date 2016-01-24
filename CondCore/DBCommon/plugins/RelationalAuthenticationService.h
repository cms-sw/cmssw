#ifndef COND_XMLAUTHENTITACTIONSERVICE_H
#define COND_XMLAUTHENTITACTIONSERVICE_H

#include "CondCore/DBCommon/interface/CredentialStore.h"
//
#include "RelationalAccess/IAuthenticationService.h"
#include "CoralKernel/Service.h"
#include "CoralKernel/Property.h"
//
#include <map>
#include <set>
#include <string>

namespace coral {

  class AuthenticationCredentials;
  //class IAuthenticationCredentials;
}

namespace cond {

  namespace RelationalAuthenticationService {

    /**
     */
    class RelationalAuthenticationService : public coral::Service,
					    virtual public coral::IAuthenticationService
    {

    public:
      /// Standard Constructor
      explicit RelationalAuthenticationService( const std::string& name );   

      /// Standard Destructor
      virtual ~RelationalAuthenticationService();

    public:

      /// Sets the input file name 
      void setAuthenticationPath(  const std::string& inputPath );

      /**
       * Returns a reference to the credentials object for a given connection string.
       * If the connection string is not known to the service an UnknownConnectionException is thrown.
       */
      const coral::IAuthenticationCredentials& credentials( const std::string& connectionString ) const;

      /**
       * Returns a reference to the credentials object for a given connection string.
       * If the connection string is not known to the service an UnknownConnectionException is thrown.
       * If the role is not known to the service an UnknownRoleException is thrown.
       */
      const coral::IAuthenticationCredentials& credentials( const std::string& connectionString,
                                                            const std::string& role ) const;

    private:

      /// The input file with the data
      std::string m_authenticationPath;

      /// The service providing the authentication data
      mutable CredentialStore m_db;

      mutable coral_bridge::AuthenticationCredentialSet m_cache;

      coral::Property::CallbackID m_callbackID;

    };

  }

}

#endif
