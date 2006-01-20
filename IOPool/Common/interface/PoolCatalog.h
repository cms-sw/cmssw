#ifndef Common_PoolCatalog_h
#define Common_PoolCatalog_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolCatalog.h,v 1.2 2005/11/23 02:17:56 wmtan Exp $
//
// Class PoolCatalog. Common services to manage POOL catalog
//
// Author: Luca Lista
// Co-Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include "FileCatalog/IFileCatalog.h"

namespace edm {

  class PoolCatalog {
  public:
    enum { WRITE = 1, READ = 2 };
    PoolCatalog(unsigned int rw, std::string url);
    ~PoolCatalog();
    void commitCatalog();
    void registerFile(std::string const& pfn, std::string const& lfn);
    void findFile(std::string & pfn, std::string const& lfn);
    static bool const isPhysical(std::string const& name) {
      return (name.empty() || name.find(':') != std::string::npos);
    }
    static std::string const toPhysical(std::string const& name) {
      return (isPhysical(name) ?  name : "file:" + name);
    }
    pool::IFileCatalog& catalog() {return catalog_;}
  private:
    pool::IFileCatalog catalog_;
  };
}

#endif
