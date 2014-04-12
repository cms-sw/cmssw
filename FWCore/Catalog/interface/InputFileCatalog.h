#ifndef FWCore_Catalog_InputFileCatalog_h
#define FWCore_Catalog_InputFileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// Class InputFileCatalog. Services to manage InputFile catalog
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include "boost/scoped_ptr.hpp"
#include "FWCore/Catalog/interface/FileLocator.h"

namespace edm {
  class FileCatalogItem {
  public:
    FileCatalogItem() : pfn_(), lfn_(), fallbackPfn_() {}
    FileCatalogItem(std::string const& pfn, std::string const& lfn, std::string const& fallbackPfn) : pfn_(pfn), lfn_(lfn), fallbackPfn_(fallbackPfn) {}
    std::string const& fileName() const {return pfn_;}
    std::string const& logicalFileName() const {return lfn_;}
    std::string const& fallbackFileName() const {return fallbackPfn_;}
  private:
    std::string pfn_;
    std::string lfn_;
    std::string fallbackPfn_;
  };

  class InputFileCatalog {
  public:
    InputFileCatalog(std::vector<std::string> const& fileNames, std::string const& override, bool useLFNasPFNifLFNnotFound = false);
    InputFileCatalog(std::vector<std::string> const& fileNames, std::string const& override, std::string const& overrideFallback, bool useLFNasPFNifLFNnotFound = false);
    ~InputFileCatalog();
    std::vector<FileCatalogItem> const& fileCatalogItems() const {return fileCatalogItems_;}
    std::vector<std::string> const& logicalFileNames() const {return logicalFileNames_;}
    std::vector<std::string> const& fileNames() const {return fileNames_;}
    std::vector<std::string> const& fallbackFileNames() const {return fallbackFileNames_;}
    bool empty() const {return fileCatalogItems_.empty();}
    static bool isPhysical(std::string const& name) {
      return (name.empty() || name.find(':') != std::string::npos);
    }
    
  private:
    void init(std::string const& override, std::string const& overrideFallback, bool useLFNasPFNifLFNnotFound);
    void findFile(std::string & pfn, std::string & fallbackPfn, std::string const& lfn, bool useLFNasPFNifLFNnotFound);
    std::vector<std::string> logicalFileNames_;
    std::vector<std::string> fileNames_;
    std::vector<std::string> fallbackFileNames_;
    std::vector<FileCatalogItem> fileCatalogItems_;
    boost::scoped_ptr<FileLocator> fileLocator_;
    boost::scoped_ptr<FileLocator> overrideFileLocator_;
    boost::scoped_ptr<FileLocator> fallbackFileLocator_;
    boost::scoped_ptr<FileLocator> overrideFallbackFileLocator_;
  };
}

#endif
