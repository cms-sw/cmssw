#ifndef CondCore_MetaDataService_METADATA_H
#define CondCore_MetaDataService_METADATA_H
#include <string>
#include <memory>
#include <vector>
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "RelationalAccess/ISession.h"
namespace coral{
  class IRelationalService;
}
namespace cond{
  class ServiceLoader;
  class MetaData {
  public:
    MetaData(const std::string& contact, ServiceLoader& loader);
    ~MetaData();
    void connect( cond::ConnectMode mod=cond::ReadWriteCreate );
    void disconnect();
    bool addMapping(const std::string& name, const std::string& token);
    bool replaceToken(const std::string& name, const std::string& newtoken);
    bool hasTag( const std::string& name ) const;
    void listAllTags( std::vector<std::string>& result ) const;
    const std::string getToken( const std::string& name );
  private:
    void createTable(const std::string& tabname);
    std::string m_con;
    coral::ISession* m_session;
    ServiceLoader& m_loader;
    coral::IRelationalService* m_service;
    cond::ConnectMode m_mode;
  };
}
#endif
