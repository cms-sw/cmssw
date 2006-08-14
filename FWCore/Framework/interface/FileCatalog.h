#ifndef Framework_FileCatalog_h
#define Framework_FileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.h,v 1.2 2006/06/07 20:29:29 wmtan Exp $
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

  class InputFileCatalog : public FileCatalog {
  public:
    explicit InputFileCatalog(ParameterSet const& pset);
    virtual ~InputFileCatalog();
    std::vector<std::string> const&  fileNames() const {
      return fileNames_;
    }
  private:
    void findFile(std::string & pfn, std::string const& lfn);
    std::vector<std::string> logicalFileNames_;
    std::vector<std::string> fileNames_;
  };

  class OutputFileCatalog : public FileCatalog {
  public:
    explicit OutputFileCatalog(ParameterSet const& pset);
    virtual ~OutputFileCatalog();
    pool::FileCatalog::FileID registerFile(std::string const& pfn, std::string const& lfn);
  };
}

#endif
