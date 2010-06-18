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
    FileCatalogItem() : pfn_(), lfn_() {}
    FileCatalogItem(std::string const& pfn, std::string const& lfn) : pfn_(pfn), lfn_(lfn) {}
    std::string const& fileName() const {return pfn_;}
    std::string const& logicalFileName() const {return lfn_;}
  private:
    std::string pfn_;
    std::string lfn_;
  };

  class InputFileCatalog {
  public:
    InputFileCatalog(std::vector<std::string> const& fileNames, std::string const& override, bool noThrow = false);
    ~InputFileCatalog();
    std::vector<FileCatalogItem> const& fileCatalogItems() const {return fileCatalogItems_;}
    std::vector<std::string> const& logicalFileNames() const {return logicalFileNames_;}
    std::vector<std::string> const& fileNames() const {return fileNames_;}
    bool empty() const {return fileCatalogItems_.empty();}
    static bool isPhysical(std::string const& name) {
      return (name.empty() || name.find(':') != std::string::npos);
    }
    
  private:
    void findFile(std::string & pfn, std::string const& lfn, bool noThrow);
    std::vector<std::string> logicalFileNames_;
    std::vector<std::string> fileNames_;
    std::vector<FileCatalogItem> fileCatalogItems_;
    boost::scoped_ptr<FileLocator> fl_;
  };
}

#endif
