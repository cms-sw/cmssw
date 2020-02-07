#ifndef FWCore_Catalog_InputFileCatalog_h
#define FWCore_Catalog_InputFileCatalog_h
//////////////////////////////////////////////////////////////////////
//
// Class InputFileCatalog. Services to manage InputFile catalog.
//
// InputFileCatalog builds FileCatalogItem, which contains PFN and LFN. The pfn_ and lfn_ correspond to the PFN and LFN constructed from the first (primary) data catalog in site-local-config.xml. The FileCatalogItem also has pfns_ which is the collection of PFNs from all data catalogs in site-local-config.xmll. Note that, the fallbackPfn_ is the PFN from the second data catalog. For backward comparability, the default setting is to use pfn_ and fallbackPfn_. Another note is that the InputFileCatalog have fileNames(), fallbackFileNames() to return PFN and fallback PFN of all input files so users can use them instead of accessing collections of FileCatalogItem (fileCatalogItems()).
//
// The multiple data catalogs mechanism is activated when initializing InputFileCatalog with setMultipleDataCatalog = true. In this case, pfns_ will be valid together with hasMultipleDataCatalogs_ = true. They can be used in RootPrimaryFileSequence, RootSecondaryFileSequence or other sequences to read files.
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
    FileCatalogItem() : pfn_(), pfns_(), lfn_(), fallbackPfn_() {}
    FileCatalogItem(std::string const& pfn, std::string const& lfn, std::string const& fallbackPfn)
        : pfn_(pfn), lfn_(lfn), fallbackPfn_(fallbackPfn) {}

    //HERE
    FileCatalogItem(std::vector<std::string> const& pfns,
                    std::string const& lfn,
                    std::string const& fallbackPfn)  //the last argument is for backward compability
        : pfns_(pfns), lfn_(lfn), fallbackPfn_(fallbackPfn) {
      if (pfns_.size() > 0)
        pfn_ = pfns_[0];
    }

    std::string const& fileName() const { return pfn_; }
    std::string const& logicalFileName() const { return lfn_; }
    std::string const& fallbackFileName() const { return fallbackPfn_; }

    //HERE
    std::vector<std::string> const& fileNames() const { return pfns_; }

  private:
    std::string pfn_;
    //HERE
    std::vector<std::string> pfns_;
    std::string lfn_;
    std::string fallbackPfn_;
  };

  class InputFileCatalog {
  public:
    //HERE
    InputFileCatalog(std::vector<std::string> const& fileNames,
                     std::string const& override,
                     bool useLFNasPFNifLFNnotFound = false,
                     bool setMultipleDataCatalog = false);
    //HERE
    InputFileCatalog(std::vector<std::string> const& fileNames,
                     std::string const& override,
                     std::string const& overrideFallback,
                     bool useLFNasPFNifLFNnotFound = false);

    ~InputFileCatalog();
    std::vector<FileCatalogItem> const& fileCatalogItems() const { return fileCatalogItems_; }
    std::vector<std::string> const& logicalFileNames() const { return logicalFileNames_; }
    std::vector<std::string> const& fileNames() const { return fileNames_; }
    std::vector<std::string> const& fallbackFileNames() const { return fallbackFileNames_; }
    bool empty() const { return fileCatalogItems_.empty(); }
    static bool isPhysical(std::string const& name) { return (name.empty() || name.find(':') != std::string::npos); }
    //HERE
    bool hasMultipleDataCatalogs() const { return hasMultipleDataCatalogs_; }

  private:
    void init(std::string const& override, std::string const& overrideFallback, bool useLFNasPFNifLFNnotFound);
    void findFile(std::string& pfn, std::string& fallbackPfn, std::string const& lfn, bool useLFNasPFNifLFNnotFound);
    //HERE
    void init(std::string const& override, bool useLFNasPFNifLFNnotFound);
    void findFile(std::string const& lfn,
                  std::vector<std::string>& pfns,
                  std::string& fallbackPfn,
                  bool useLFNasPFNifLFNnotFound);
    std::vector<std::string> logicalFileNames_;
    std::vector<std::string> fileNames_;
    std::vector<std::string> fallbackFileNames_;
    std::vector<FileCatalogItem> fileCatalogItems_;
    edm::propagate_const<std::unique_ptr<FileLocator>> fileLocator_;
    edm::propagate_const<std::unique_ptr<FileLocator>> overrideFileLocator_;
    edm::propagate_const<std::unique_ptr<FileLocator>> fallbackFileLocator_;
    edm::propagate_const<std::unique_ptr<FileLocator>> overrideFallbackFileLocator_;

    //HERE
    std::vector<edm::propagate_const<std::unique_ptr<FileLocator>>> fileLocators_;
    bool hasMultipleDataCatalogs_;
  };
}  // namespace edm

#endif
