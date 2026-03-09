#include <filesystem>

#include <boost/algorithm/string.hpp>

#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {

  InputFileCatalog::InputFileCatalog(std::vector<std::string> logicalFileNames,
                                     std::string const& override,
                                     bool useLFNasPFNifLFNnotFound,
                                     SciTagCategory sciTagCategory)
      : sciTagCategory_(sciTagCategory) {
    Service<SiteLocalConfig> localconfservice;
    if (!localconfservice.isAvailable()) {
      cms::Exception ex("FileCatalog");
      ex << "edm::SiteLocalConfigService is not available";
      ex.addContext("Calling edm::InputFileCatalog constructor");
      throw ex;
    }

    if (!overrideFileLocator_ && !override.empty()) {
      //now make a struct from input string
      std::vector<std::string> tmps;
      boost::algorithm::split(tmps, override, boost::is_any_of(std::string(",")));
      if (tmps.size() != 5) {
        cms::Exception ex("FileCatalog");
        ex << "Trying to override input file catalog but invalid input override string " << override
           << " (Should be site,subSite,storageSite,volume,protocol)";
        ex.addContext("Calling edm::InputFileCatalog constructor");
        throw ex;
      }

      CatalogAttributes override_struct(tmps[0],   //current-site
                                        tmps[1],   //current-subSite
                                        tmps[2],   //desired-data-access-site
                                        tmps[3],   //desired-data-access-volume
                                        tmps[4]);  //desired-data-access-protocol
      if (override_struct.empty()) {
        cms::Exception ex("FileCatalog");
        ex << "Trying to override input file catalog but invalid input override string " << override
           << "\nResulting CatalogAttributes is empty (should be site,subSite,storageSite,volume,protocol)";
        ex.addContext("Calling edm::InputFileCatalog constructor");
        throw ex;
      }
      std::filesystem::path filename_storage = localconfservice->storageDescriptionPath(override_struct);
      overrideFileLocator_ = std::make_unique<FileLocator>(override_struct, filename_storage);
    } else {
      std::vector<CatalogAttributes> const& tmp_dataCatalogs = localconfservice->dataCatalogs();
      fileLocators_.clear();
      // Construct all file locators from data catalogs. If a data catalog is
      // invalid (wrong protocol for example), it is skipped and no file locator
      // is constructed (an exception is thrown out from the FileLocator constructor).
      for (const auto& catalogAttributes : tmp_dataCatalogs) {
        if (catalogAttributes.empty()) {
          edm::LogWarning("InputFileCatalog")
              << "Empty CatalogAttributes object in InputFileCatalog constructor. This catalog will be skipped.";
          continue;
        }
        try {
          std::filesystem::path filename_storage = localconfservice->storageDescriptionPath(catalogAttributes);
          fileLocators_.push_back(std::make_unique<FileLocator>(catalogAttributes, filename_storage));
        } catch (cms::Exception const& e) {
          edm::LogWarning("InputFileCatalog")
              << "Caught an exception while constructing a file locator in InputFileCatalog constructor: " << e.what()
              << "Skip this catalog";
        }
      }
      if (fileLocators_.empty()) {
        cms::Exception ex("FileCatalog");
        ex << "Unable to construct any file locator in InputFileCatalog constructor";
        ex.addContext("Calling edm::InputFileCatalog constructor");
        throw ex;
      }
    }

    fileCatalogItems_.reserve(logicalFileNames.size());
    for (auto& lfn : logicalFileNames) {
      boost::trim(lfn);
      std::vector<std::string> pfns;
      if (lfn.empty()) {
        cms::Exception ex("FileCatalog");
        ex << "An empty string specified in the fileNames parameter for input source";
        ex.addContext("Calling edm::InputFileCatalog constructor");
        throw ex;
      }
      if (isPhysical(lfn)) {
        if (lfn.back() == ':') {
          cms::Exception ex("FileCatalog");
          ex << "An empty physical file name specified in the fileNames parameter for input source";
          ex.addContext("Calling edm::InputFileCatalog constructor");
          throw ex;
        }
        pfns.push_back(lfn);
        // Clear the LFN.
        lfn.clear();
      } else {
        findFile(lfn, pfns, useLFNasPFNifLFNnotFound);
      }
      lfn.shrink_to_fit();  // try to release memory

      if (sciTagCategory_ != SciTagCategory::Undefined) {
        Service<StorageURLModifier> sciTagConfigService;
        if (!sciTagConfigService.isAvailable()) {
          cms::Exception ex("FileCatalog");
          ex << "edm::SciTagConfig service is not available";
          ex.addContext("Calling edm::InputFileCatalog constructor");
          throw ex;
        }

        for (auto& pfn : pfns) {
          sciTagConfigService->modify(sciTagCategory_, pfn);
        }
      }

      fileCatalogItems_.emplace_back(std::move(pfns), std::move(lfn));
    }
  }

  InputFileCatalog::~InputFileCatalog() = default;

  std::vector<std::string> InputFileCatalog::fileNames(unsigned iCatalog) const {
    std::vector<std::string> tmp;
    tmp.reserve(fileCatalogItems_.size());
    for (auto const& item : fileCatalogItems_) {
      tmp.push_back(item.fileName(iCatalog));
    }
    return tmp;
  }

  void InputFileCatalog::findFile(std::string const& lfn,
                                  std::vector<std::string>& pfns,
                                  bool useLFNasPFNifLFNnotFound) const {
    if (overrideFileLocator_) {
      pfns.emplace_back(overrideFileLocator_->pfn(lfn));
    } else {
      for (auto const& locator : fileLocators_) {
        std::string pfn = locator->pfn(lfn);
        if (pfn.empty() && useLFNasPFNifLFNnotFound) {
          pfns.emplace_back(lfn);
        } else {
          pfns.emplace_back(std::move(pfn));
        }
      }
    }

    // Empty PFN will be found by caller.
  }

}  // namespace edm
