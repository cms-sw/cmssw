#ifndef INCLUDE_COND_CREDENTIALSTORE_H
#define INCLUDE_COND_CREDENTAILSTORE_H

#include "CondCore/DBCommon/interface/DecodingKey.h"
//
#include <map>
#include <string>
//#include <memory>
#include <boost/shared_ptr.hpp>
//

namespace coral {

  class AuthenticationCredentials;
  class IAuthenticationCredentials;
  class ISession;
  class IConnection;

}

namespace coral_bridge {

  class AuthenticationCredentialSet
  {
  public:

    static const std::string DEFAULT_ROLE;
  public:
    /// Constructor
    AuthenticationCredentialSet();

    /// Destructor
    virtual ~AuthenticationCredentialSet();


    void registerItem( const std::string& connectionString, 
		       const std::string& itemName,
		       const std::string& itemValue );
			      

    void registerItem( const std::string& connectionString, 
		       const std::string& role,
		       const std::string& itemName,
		       const std::string& itemValue );

    /**
     * Adds a credential item to the default role.
     */
    void registerCredentials( const std::string& connectionString,
			      const std::string& userName,
			      const std::string& password );

    /**
     * Adds a credential item to the specified role.
     */
    void registerCredentials( const std::string& connectionString,
			      const std::string& role,
			      const std::string& userName,
			      const std::string& password );

    void import( const AuthenticationCredentialSet& data );

    const coral::IAuthenticationCredentials* get( const std::string& connectionString ) const;

    const coral::IAuthenticationCredentials* get( const std::string& connectionString, const std::string& role ) const;

    const std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* >& data() const ;

  private:
    /// credentials for the specific roles 
    std::map< std::pair<std::string,std::string>, coral::AuthenticationCredentials* > m_data;

  };

}

namespace cond {

  class Cipher;

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
    void setUpForService( const std::string& serviceName );

    void setUpForConnectionString( const std::string& connectionString );
    
    bool createSchema( );

    bool drop( );

    /// add a credential entry into the repository
    bool update( const std::string& principal, 
		 const std::string& role,
		 const std::string& connectionString,  
		 const std::string& userName, const std::string& password );
    
    /// remove a credential entry from the repository
    bool remove( const std::string& principal, 
		 const std::string& role,
		 const std::string& connectionString );
    
    /// remove a credential entry from the repository
    bool removePrincipal( const std::string& principal );

    bool exportForPrincipal( const std::string& principal, coral_bridge::AuthenticationCredentialSet& destinationData );
    
    /// import/export data 
    bool importForPrincipal( const std::string& principal, const coral_bridge::AuthenticationCredentialSet& data );    

    bool exportAll( coral_bridge::AuthenticationCredentialSet& destinationData );
    
    private:
    struct CoralSession {
      boost::shared_ptr<coral::IConnection> connection;
      boost::shared_ptr<coral::ISession> session;
      ~CoralSession();
    };

    CoralSession openDatabase( bool readMode );
    
    private:

    const ServiceKey* m_serviceKey;

    /// The cipher
    DecodingKey m_key;


  };

}




#endif
