#ifndef DBCommon_DBCatalog_h
#define DBCommon_DBCatalog_h
#include <string>
#include <map>
namespace pool{
  class IFileCatalog;
}
namespace cond{
  class DBCatalog{
  public:
    static std::string defaultOnlineCatalogName();
    static std::string defaultOfflineCatalogName();
    static std::string defaultDevCatalogName();
    static std::string defaultLocalCatalogName();
    /// constructor
    DBCatalog();
    /// destructor
    ~DBCatalog();
    /// get pool filecatalog handle
    pool::IFileCatalog& poolCatalog();
    /// if a string is not in lfn format, return empty string
    std::pair<std::string,std::string> 
      logicalservice( const std::string& input ) const;
    /// if a string is in lfn format
    //bool isLFN(const std::string& input) const;
    /// looks up PFN with given LFN. If useCache is true, select pfn contains frontier protocol. This operation involves catalog transaction
    std::string getPFN(pool::IFileCatalog& poolCatalog,
			     const std::string& lfn, 
			     bool useCache=false);
  private:
    pool::IFileCatalog* m_poolcatalog;
  };//cl DBCatalog
}//ns cond
#endif
