#ifndef CondCoreDBCommon_DecodingKey_H
#define CondCoreDBCommon_DecodingKey_H

#include <iostream>
#include <string>
#include <set>
#include <map>

namespace cond {

  struct ServiceCredentials {
    ServiceCredentials();
    std::string connectionString;
    std::string userName;
    std::string password;
  };

  class KeyGenerator {
    public:

    KeyGenerator();

    std::string make( size_t keySize );
    std::string makeWithRandomSize( size_t maxSize );
    
    private:

    int m_iteration;

  };

  class DecodingKey {

    public:

    static const std::string FILE_NAME;
    static const std::string FILE_PATH;
    static const size_t DEFAULT_KEY_SIZE = 100;
    static std::string templateFile();

    public:

    DecodingKey();

    virtual ~DecodingKey(){}

    size_t init( const std::string& keyFileName,  const std::string& password, bool readMode = true );

    size_t createFromInputFile( const std::string& inputFileName, size_t generatedKeySize = 0 );

    void list( std::ostream& out );

    void flush();

    const std::string& principalName() const;

    const std::string& principalKey() const;

    bool isNominal() const;

    const std::string& ownerName() const;

    const std::map< std::string, ServiceCredentials >& services() const;

    void addDefaultService( const std::string& connectionString );

    void addService( const std::string& serviceName, const std::string& connectionString, const std::string& userName, const std::string& password );


    private:

    std::string m_fileName;

    bool m_mode;

    std::string m_pwd;

    std::string m_principalName;

    std::string m_principalKey;

    std::string m_owner;

    std::map< std::string, ServiceCredentials > m_services;
    
  };
}

inline
cond::KeyGenerator::KeyGenerator():m_iteration(0){
}

inline
cond::ServiceCredentials::ServiceCredentials():connectionString(""),userName(""),password(""){
}

inline
cond::DecodingKey::DecodingKey():m_fileName(""),m_mode( true ),m_pwd(""),m_principalName(""),m_principalKey(""),m_owner(""),m_services(){
}

inline
const std::string& 
cond::DecodingKey::principalName() const {
  return m_principalName;
}

inline
const std::string& 
cond::DecodingKey::principalKey() const {
  return  m_principalKey;
}
 
inline
bool 
cond::DecodingKey::isNominal() const {
  return  !m_owner.empty();
}

inline
const std::string& 
cond::DecodingKey::ownerName() const {
  return  m_owner;
}

inline
const std::map< std::string, cond::ServiceCredentials >&
cond::DecodingKey::services() const { return m_services; }

#endif //  CondCoreDBCommon_DecodingKey_H

  
