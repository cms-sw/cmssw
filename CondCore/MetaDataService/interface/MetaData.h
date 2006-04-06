#ifndef COND_METADATA_H
#define COND_METADATA_H
#include <string>
#include <memory>
#include <vector>
#include "RelationalAccess/ISession.h"
//namespace coral{
  //class ITable;
//}
namespace cond{
  class ServiceLoader;
  class MetaData {
  public:
    MetaData(const std::string& contact, ServiceLoader& loader);
    ~MetaData();
    void connect();
    void disconnect();
    bool addMapping(const std::string& name, const std::string& token);
    bool replaceToken(const std::string& name, const std::string& newtoken);
    bool hasTag( const std::string& name ) const;
    void listAllTags( std::vector<std::string>& result ) const;
    const std::string getToken( const std::string& name );
  private:
    void createTable(const std::string& tabname);
    std::string m_con;
    std::auto_ptr< coral::ISession > m_session;
    ServiceLoader& m_loader;
  };
}
#endif
