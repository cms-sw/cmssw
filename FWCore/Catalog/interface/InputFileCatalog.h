#ifndef FWCore_Catalog_InputFileCatalog_h
#define FWCore_Catalog_InputFileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// Class InputFileCatalog. Services to manage InputFile catalog.
// Physical file names, pfns_ of FileCatalogItem, are constructed from multiple data catalogs in site-local-config.xml. Each member of pfns_ corresponds to a data catalog.
// Note that fileNames(unsigned iCatalog) of InputFileCatalog return physical file names of all input files corresponding to a data catalog (for example, a job has 10 input files provided as a PoolSource, the fileNames(unsigned iCatalog) will return PFNs of these 10 files constructed from a data catalog)
// Set catType=TrivialCatalog: use trivial data catalogs from <event-data>
// Set catType=RucioCatalog: use data catalogs from <data-access> and storage.json
//
//////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <vector>
#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace edm {
  class FileCatalogItem {
  public:
    FileCatalogItem(std::vector<std::string> const& pfns, std::string const& lfn) : pfns_(pfns), lfn_(lfn) {}

    std::string const& fileName(unsigned iCatalog) const { return pfns_[iCatalog]; }
    std::string const& logicalFileName() const { return lfn_; }

    std::vector<std::string> const& fileNames() const { return pfns_; }

  private:
    std::vector<std::string> pfns_;
    std::string lfn_;
  };

  class InputFileCatalog {
  public:
    InputFileCatalog(std::vector<std::string> const& fileNames,
                     std::string const& override,
                     bool useLFNasPFNifLFNnotFound = false,
                     //switching between two catalog types
                     //edm::CatalogType catType = edm::CatalogType::TrivialCatalog);
                     edm::CatalogType catType = edm::CatalogType::RucioCatalog);

    ~InputFileCatalog();
    std::vector<FileCatalogItem> const& fileCatalogItems() const { return fileCatalogItems_; }
    std::vector<std::string> const& logicalFileNames() const { return logicalFileNames_; }
    std::vector<std::string> fileNames(unsigned iCatalog) const;
    bool empty() const { return fileCatalogItems_.empty(); }
    static bool isPhysical(std::string const& name) { return (name.empty() || name.find(':') != std::string::npos); }

  private:
    void init(std::string const& override, bool useLFNasPFNifLFNnotFound, edm::CatalogType catType);
    void findFile(std::string const& lfn,
                  std::vector<std::string>& pfns,
                  bool useLFNasPFNifLFNnotFound,
                  edm::CatalogType catType);
    std::vector<std::string> logicalFileNames_;
    std::vector<std::string> fileNames_;
    std::vector<FileCatalogItem> fileCatalogItems_;
    edm::propagate_const<std::unique_ptr<FileLocator>> overrideFileLocator_;

    std::vector<edm::propagate_const<std::unique_ptr<FileLocator>>> fileLocators_trivalCatalog_;
    std::vector<edm::propagate_const<std::unique_ptr<FileLocator>>> fileLocators_;
  };
}  // namespace edm

#endif
