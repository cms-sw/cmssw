#ifndef CondCore_MetaDataService_METADATA_H
#define CondCore_MetaDataService_METADATA_H
#include <string>
#include <vector>
namespace cond{
  class CoralTransaction;
  class MetaData {
  public:
    explicit MetaData( cond::CoralTransaction& coraldb );
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
    cond::CoralTransaction& m_coraldb;
  };
}
#endif
 
