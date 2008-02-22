#ifndef FWCore_Catalog_InputFileCatalog_h
#define FWCore_Catalog_InputFileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// $Id: InputFileCatalog.h,v 1.1 2007/03/04 04:43:30 wmtan Exp $
//
// Class InputFileCatalog. Services to manage InputFile catalog
//
// Author of original version: Luca Lista
// Author of current version: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include "FWCore/Catalog/interface/FileCatalog.h"

namespace edm {
  class ParameterSet;
  class InputFileCatalog : public FileCatalog {
  public:
    explicit InputFileCatalog(ParameterSet const& pset, bool noThrow = false);
    virtual ~InputFileCatalog();
    std::vector<FileCatalogItem> const& fileCatalogItems() const {return fileCatalogItems_;}
    std::vector<std::string> const& logicalFileNames() const {return logicalFileNames_;}
    std::vector<std::string> const& fileNames() const {return fileNames_;}
  private:
    void findFile(std::string & pfn, std::string const& lfn, bool noThrow);
    std::vector<std::string> logicalFileNames_;
    std::vector<std::string> fileNames_;
    std::vector<FileCatalogItem> fileCatalogItems_;
  };
}

#endif
