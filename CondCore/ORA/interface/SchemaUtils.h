#ifndef INCLUDE_ORA_SCHEMAUTILS_H
#define INCLUDE_ORA_SCHEMAUTILS_H

#include <string>
#include <set>
#include <memory>

namespace coral {
  class ConnectionService;
  class ISessionProxy;
}

namespace ora {

  namespace SchemaUtils {

    void cleanUp( const std::string& connectionString, std::set<std::string> exclusionList=std::set<std::string>() );

  }  

  class Serializer {
  public:
    static const std::string& tableName();

  public:
    Serializer();
      
    virtual ~Serializer();
    
    void lock( const std::string& connectionString );

    void release();

  private:

    std::auto_ptr<coral::ConnectionService> m_connServ;
    std::auto_ptr<coral::ISessionProxy> m_session;
    bool m_lock;
    
  };

}

#endif


