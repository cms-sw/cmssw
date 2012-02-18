#ifndef CondCoreDBCommon_DecodingKey_H
#define CondCoreDBCommon_DecodingKey_H

#include <iostream>
#include <string>
#include <set>
#include <map>

namespace cond {

  struct ServiceKey {
    ServiceKey();
    std::string dataSource;
    std::string key;
    std::string userName;
    std::string password;
  };

  class DecodingKey {

    public:

    static const std::string FILE_NAME;

    public:

    DecodingKey();

    virtual ~DecodingKey(){}

    size_t init( const std::string& keyFileName,  const std::string& password, bool readMode = true );

    size_t createFromInputFile( const std::string& inputFileName, bool generateKey = true );

    void list( std::ostream& out );

    void flush();

    const std::string& user() const;
    const std::set<std::string>& groups() const;
    const std::map< std::string, ServiceKey >& serviceKeys() const;

    void setUser( const std::string& user );

    void addGroup( const std::string& group );

    void addKeyForDefaultService( const std::string& dataSource, const std::string& key );

    void addDefaultService( const std::string& dataSource );

    void addKeyForService( const std::string& serviceName, const std::string& dataSource, const std::string& key, const std::string& userName, const std::string& password );

    void addService( const std::string& serviceName, const std::string& dataSource, const std::string& userName, const std::string& password );

    private:
    
    std::string generateKey();

    private:

    std::string m_fileName;

    bool m_mode;

    int m_iteration;

    std::string m_pwd;

    std::string m_user;

    std::set<std::string> m_groups;

    std::map< std::string, ServiceKey > m_serviceKeys;
    
  };
}

inline
cond::ServiceKey::ServiceKey():dataSource(""),key(""),userName(""),password(""){
}

inline
cond::DecodingKey::DecodingKey():m_fileName(""),m_mode( true ),m_iteration(0),m_pwd(""),m_user(""),m_groups(),m_serviceKeys(){
}

inline
const std::string& 
cond::DecodingKey::user() const {
  return m_user;
}

inline
const std::set< std::string >& 
cond::DecodingKey::groups() const {
  return m_groups;
}

inline
const std::map< std::string, cond::ServiceKey >&
cond::DecodingKey::serviceKeys() const { return m_serviceKeys; }

#endif //  CondCoreDBCommon_DecodingKey_H

  
