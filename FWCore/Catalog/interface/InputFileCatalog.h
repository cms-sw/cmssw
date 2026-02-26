#ifndef FWCore_Catalog_InputFileCatalog_h
#define FWCore_Catalog_InputFileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// Class InputFileCatalog.
//
// The physical file names, pfns_ of each FileCatalogItem, are constructed from
// multiple data catalogs in site-local-config.xml. Each member of pfns_ corresponds
// to a data catalog.
//
// Note that InputFileCatalog::fileNames(unsigned iCatalog) returns the physical file
// names of all input files corresponding to a data catalog (for example, a job has
// 10 input files provided via a PoolSource, InputFileCatalog::fileNames(unsigned iCatalog)
// will return the PFNs of those 10 files constructed from one data catalog)
//
// Catalogs are based on Rucio (from <data-access> and storage.json)
//
// Note that support for TrivialFileCatalog was removed in 2025.
//
//////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FWCore/Catalog/interface/StorageURLModifier.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace edm {

  class FileLocator;

  class FileCatalogItem {
  public:
    FileCatalogItem(std::vector<std::string> pfns, std::string lfn) : pfns_(std::move(pfns)), lfn_(std::move(lfn)) {}

    std::string const& fileName(unsigned iCatalog) const { return pfns_[iCatalog]; }
    std::string const& logicalFileName() const { return lfn_; }
    std::vector<std::string> const& fileNames() const { return pfns_; }

  private:
    std::vector<std::string> pfns_;
    std::string lfn_;
  };

  class InputFileCatalog {
  public:
    InputFileCatalog(std::vector<std::string> logicalFileNames,
                     std::string const& override,
                     bool useLFNasPFNifLFNnotFound = false,
                     SciTagCategory sciTagCategory = SciTagCategory::Primary);
    ~InputFileCatalog();

    std::vector<FileCatalogItem> const& fileCatalogItems() const { return fileCatalogItems_; }
    std::vector<std::string> fileNames(unsigned iCatalog) const;
    bool empty() const { return fileCatalogItems_.empty(); }
    static bool isPhysical(std::string const& name) { return (name.empty() || name.find(':') != std::string::npos); }

  private:
    void findFile(std::string const& lfn, std::vector<std::string>& pfns, bool useLFNasPFNifLFNnotFound) const;

    std::vector<FileCatalogItem> fileCatalogItems_;
    propagate_const<std::unique_ptr<FileLocator>> overrideFileLocator_;
    std::vector<propagate_const<std::unique_ptr<FileLocator>>> fileLocators_;
    SciTagCategory sciTagCategory_;
  };
}  // namespace edm
#endif
