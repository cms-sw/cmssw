#ifndef FWCore_Catalog_FileCatalog_h
#define FWCore_Catalog_FileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.h,v 1.1 2007/03/04 04:43:30 wmtan Exp $
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
    FileCatalogItem() : pfn_(0), lfn_(0) {}
    FileCatalogItem(std::string const& pfn, std::string const& lfn) : pfn_(&pfn), lfn_(&lfn) {}
    std::string const& fileName() const {return *pfn_;}
    std::string const& logicalFileName() const {return *lfn_;}
  private:
    std::string const* pfn_;
    std::string const* lfn_;
  };

  class ParameterSet;
  class FileCatalog {
  public:
    explicit FileCatalog(ParameterSet const& pset);
    virtual ~FileCatalog() = 0;
    void commitCatalog();
    static bool const isPhysical(std::string const& name) {
      return (name.empty() || name.find(':') != std::string::npos);
    }
    static std::string const toPhysical(std::string const& name) {
      return (isPhysical(name) ?  name : "file:" + name);
    }
    pool::IFileCatalog& catalog() {return catalog_;}
    std::string& url() {return url_;}
    void setActive() {active_ = true;}
    bool active() const {return active_;}
  private:
    pool::IFileCatalog catalog_;
    std::string url_;
    bool active_;
  };
}

#endif
