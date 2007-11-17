#ifndef COND_DBCommon_ObjectRelationalMappingUtility_h
#define COND_DBCommon_ObjectRelationalMappingUtility_h
#include <string>
#include <vector>
namespace pool{
  class ObjectRelationalMappingUtilities;
}

namespace cond{
  class RelationalStorageManager;
  class ObjectRelationalMappingUtility{
  public:
    explicit ObjectRelationalMappingUtility( cond::RelationalStorageManager& coraldb );
    ~ObjectRelationalMappingUtility();

    void buildAndStoreMappingFromBuffer( const std::string& buffer );
    
    void listMappings( std::vector<std::string>& mappinglist );

    bool existsMapping(const std::string& version);

    void removeMapping(const std::string& version, 
		       bool removeDataTables=false);
  private:
    pool::ObjectRelationalMappingUtilities* m_mappingutil;
  };
}
#endif
