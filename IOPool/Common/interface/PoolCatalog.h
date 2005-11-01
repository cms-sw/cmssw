#ifndef Common_PoolCatalog_h
#define Common_PoolCatalog_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolCatalog.h,v 1.1 2005/07/27 12:37:32 wmtan Exp $
//
// Class PoolCatalog. Common services to manage POOL catalog and cache
//
// Author: Luca Lista
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include "FileCatalog/IFileCatalog.h"

namespace pool {
  class IDataSvc;
}

namespace edm {

  class PoolCatalog {
  public:
    enum { WRITE = 1, READ = 2 };
    PoolCatalog(unsigned int rw, std::string const& url = std::string());
    ~PoolCatalog();
    pool::IDataSvc * createContext(bool write, bool delete_on_free);
    void commitCatalog();
    void registerFile(std::string const& pfn, std::string const& lfn);
    void findFile(std::string & pfn, std::string const& lfn);
    static bool const isPhysical(std::string const& name) {
      return (name.empty() || name.find(':') != std::string::npos);
    }
    static std::string const toPhysical(std::string const& name) {
      return (isPhysical(name) ?  name : "file:" + name);
    }
  private:
    pool::IFileCatalog catalog_;
  };
}

#endif
