#ifndef CondCore_MetaDataService_METADATA_H
#define CondCore_MetaDataService_METADATA_H
#include <string>
#include <vector>
namespace coral{
  class ISessionProxy;
}
namespace cond{
  class RelationalStorageManager;
  class MetaData {
  public:
    explicit MetaData( cond::RelationalStorageManager& coraldb );
    ~MetaData();
    bool addMapping(const std::string& name, const std::string& token);
    bool replaceToken(const std::string& name, const std::string& newtoken);
    bool hasTag( const std::string& name ) const;
    void listAllTags( std::vector<std::string>& result ) const;
    const std::string getToken( const std::string& name );
    void deleteAllEntries();
    void deleteEntryByToken( const std::string& token );
    void deleteEntryByTag( const std::string& tag );
  private:
    void createTable(const std::string& tabname);
    cond::RelationalStorageManager& m_coraldb;
    coral::ISessionProxy* m_proxy;
  };
}
#endif
