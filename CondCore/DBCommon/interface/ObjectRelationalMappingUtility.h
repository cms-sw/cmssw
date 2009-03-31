#ifndef COND_DBCommon_ObjectRelationalMappingUtility_h
#define COND_DBCommon_ObjectRelationalMappingUtility_h
#include <string>
#include <vector>
namespace pool{
  class ObjectRelationalMappingUtilities;
}
namespace coral{
  class ISessionProxy;
}
namespace cond{
  class RelationalStorageManager;
  class ObjectRelationalMappingUtility{
  public:
    explicit ObjectRelationalMappingUtility( coral::ISessionProxy*  );
    ~ObjectRelationalMappingUtility();

    void buildAndStoreMappingFromBuffer( const std::string& buffer );

    void buildAndStoreMappingFromFile( const std::string& filename );

    //void listMappings( std::vector<std::string>& mappinglist );

    bool existsMapping(const std::string& version);
    
    void removeMapping(const std::string& version,bool removeTables=true);

    //copy mapping from default session to session for one container
    bool exportMapping(coral::ISessionProxy* session, 
		       std::string const & contName, std::string const & classVersion, 
		       bool allVersions=false);

  private:
    pool::ObjectRelationalMappingUtilities* m_mappingutil;
    coral::ISessionProxy* m_coralsessionHandle;
  };
}
#endif
