#ifndef FWCore_Catalog_InputFileCatalog_h
#define FWCore_Catalog_InputFileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// Class InputFileCatalog. Services to manage InputFile catalog
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include "FWCore/Catalog/interface/FileCatalog.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
  
  class InputFileCatalog : public FileCatalog {
  public:
    explicit InputFileCatalog(ParameterSet const& pset,
			      std::string const& namesParameter = std::string("fileNames"),
			      bool canBeEmpty = false,
			      bool noThrow = false);
    virtual ~InputFileCatalog();
    std::vector<FileCatalogItem> const& fileCatalogItems() const {return fileCatalogItems_;}
    std::vector<std::string> const& logicalFileNames() const {return logicalFileNames_;}
    std::vector<std::string> const& fileNames() const {return fileNames_;}
    bool empty() const {return fileCatalogItems_.empty();}
    
    static void fillDescription(ParameterSetDescription & desc);
    
  private:
    void findFile(std::string & pfn, std::string const& lfn, bool noThrow);
    std::vector<std::string> logicalFileNames_;
    std::vector<std::string> fileNames_;
    std::vector<FileCatalogItem> fileCatalogItems_;
  };
}

#endif
