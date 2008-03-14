#ifndef FWCore_Catalog_FileCatalog_h
#define FWCore_Catalog_FileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.h,v 1.4 2008/02/22 19:08:02 wmtan Exp $
//
// Class FileCatalog. Common services to manage File catalog
//
// Author of original version: Luca Lista
// Author of current version: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include "FileCatalog/IFileCatalog.h"

namespace edm {

  class FileCatalogItem {
  public:
    FileCatalogItem() : pfn_(), lfn_() {}
    FileCatalogItem(std::string const& pfn, std::string const& lfn) : pfn_(pfn), lfn_(lfn) {}
    std::string const& fileName() const {return pfn_;}
    std::string const& logicalFileName() const {return lfn_;}
  private:
    std::string pfn_;
    std::string lfn_;
  };

  struct PoolCatalog {
    PoolCatalog() : catalog_() {}
    pool::IFileCatalog catalog_;
  };

  class ParameterSet;
  class FileCatalog {
  public:
    explicit FileCatalog(PoolCatalog & poolcat);
    virtual ~FileCatalog() = 0;
    void commitCatalog();
    static bool const isPhysical(std::string const& name) {
      return (name.empty() || name.find(':') != std::string::npos);
    }
    pool::IFileCatalog& catalog() {return catalog_;}
    std::string & url() {return url_;}
    std::string const& url() const {return url_;}
    void setActive() {active_ = true;}
    bool active() const {return active_;}
  private:
    pool::IFileCatalog& catalog_;
    std::string url_;
    bool active_;
  };
}

#endif
