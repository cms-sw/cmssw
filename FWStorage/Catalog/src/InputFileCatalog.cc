#include <filesystem>
#include <ranges>
#include <utility>

#include <boost/algorithm/string.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWStorage/Catalog/interface/FileLocator.h"
#include "FWStorage/Catalog/interface/InputFileCatalog.h"
#include "FWStorage/Catalog/interface/SiteLocalConfig.h"

namespace {
  std::string const emptyString;
}

namespace edm {
  InputFileCatalog::InputFileCatalog(ParameterSet const& pset,
                                     bool useLFNasPFNifLFNnotFound,
                                     SciTagCategory sciTagCategory)
      : InputFileCatalog(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
                         pset.getUntrackedParameter<std::string>("overrideCatalog", std::string()),
                         useLFNasPFNifLFNnotFound,
                         sciTagCategory) {}

  InputFileCatalog::InputFileCatalog(std::vector<std::string> configuredFileNames,
                                     std::string const& override,
                                     bool useLFNasPFNifLFNnotFound,
                                     SciTagCategory sciTagCategory)
      : configuredFileNames_(std::move(configuredFileNames)),
        sciTagCategory_(sciTagCategory),
        useLFNasPFNifLFNnotFound_(useLFNasPFNifLFNnotFound) {
    Service<SiteLocalConfig> localconfservice;
    if (!localconfservice.isAvailable()) {
      cms::Exception ex("FileCatalog");
      ex << "edm::SiteLocalConfigService is not available";
      ex.addContext("Calling edm::InputFileCatalog constructor");
      throw ex;
    }

    if (!override.empty()) {
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

    if (!configuredFileNames_.empty() && sciTagCategory_ != SciTagCategory::Undefined) {
      Service<StorageURLModifier> sciTagConfigService;
      if (!sciTagConfigService.isAvailable()) {
        cms::Exception ex("FileCatalog");
        ex << "edm::SciTagConfig service is not available";
        ex.addContext("Calling edm::InputFileCatalog constructor");
        throw ex;
      }
    }

    for (auto& configuredFileName : configuredFileNames_) {
      boost::trim(configuredFileName);
      if (configuredFileName.empty()) {
        cms::Exception ex("FileCatalog");
        ex << "An empty string specified in the fileNames parameter for input source";
        ex.addContext("Calling edm::InputFileCatalog constructor");
        throw ex;
      }
      if (isPhysicalFileName(configuredFileName)) {
        if (configuredFileName.back() == ':') {
          cms::Exception ex("FileCatalog");
          ex << "An empty physical file name specified in the fileNames parameter for input source";
          ex.addContext("Calling edm::InputFileCatalog constructor");
          throw ex;
        }
      }
      configuredFileName.shrink_to_fit();  // try to release memory
    }
  }

  InputFileCatalog::~InputFileCatalog() = default;

  void InputFileCatalog::fillDescription(ParameterSetDescription& desc) {
    desc.addUntracked<std::vector<std::string> >("fileNames")->setComment("Names of files to be processed.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
  }

  bool InputFileCatalog::isPhysicalFileName(std::string const& configuredFileName) {
    return configuredFileName.find(':') != std::string::npos;
  }

  std::string const& InputFileCatalog::logicalFileName(unsigned int fileIndex) const {
    if (fileIndex >= configuredFileNames_.size()) {
      cms::Exception ex("FileCatalog");
      ex << "Out of range argument";
      ex.addContext("Calling edm::InputFileCatalog::logicalFileName with fileIndex");
      throw ex;
    }
    return logicalFileName(configuredFileNames_[fileIndex]);
  }

  std::string const& InputFileCatalog::logicalFileName(std::string const& configuredFileName) const {
    if (isPhysicalFileName(configuredFileName)) {
      return emptyString;
    } else {
      return configuredFileName;
    }
  }

  std::vector<std::string> InputFileCatalog::physicalFileNames(std::string const& configuredFileName) const {
    std::vector<std::string> pfns;

    bool isPFN = isPhysicalFileName(configuredFileName);
    unsigned int numberOfPFNs = (isPFN || overrideFileLocator_) ? 1 : fileLocators_.size();
    pfns.reserve(numberOfPFNs);
    for (unsigned int iCatalog : std::views::iota(0u, numberOfPFNs)) {
      pfns.push_back(physicalFileName(iCatalog, configuredFileName, isPFN));
    }
    return pfns;
  }

  std::vector<std::string> InputFileCatalog::physicalFileNames(unsigned int fileIndex) const {
    if (fileIndex >= configuredFileNames_.size()) {
      cms::Exception ex("FileCatalog");
      ex << "Out of range argument";
      ex.addContext("Calling edm::InputFileCatalog::physicalFileNames with fileIndex");
      throw ex;
    }
    return physicalFileNames(configuredFileNames_[fileIndex]);
  }

  std::string InputFileCatalog::firstPFNFromFirstCatalog() const {
    const unsigned int iCatalog = 0;
    if (configuredFileNames_.empty()) {
      // If using this function, there should have been at least
      // one configured file name passed in to the InputFileCatalog
      // constructor. The client of InputFileCatalog must ensure that
      // before using this function.
      cms::Exception ex("FileCatalog");
      ex << "No file names configured";
      ex.addContext("Calling edm::InputFileCatalog::firstPFNFromFirstCatalog");
      throw ex;
    }
    std::string const& firstConfiguredFileName = configuredFileNames_[0];
    bool isPFN = isPhysicalFileName(firstConfiguredFileName);
    return physicalFileName(iCatalog, firstConfiguredFileName, isPFN);
  }

  std::vector<std::string> InputFileCatalog::allPFNsFromFirstCatalog() const {
    const unsigned int iCatalog = 0;
    std::vector<std::string> pfns;
    for (auto const& configuredFileName : configuredFileNames_) {
      bool isPFN = isPhysicalFileName(configuredFileName);
      pfns.push_back(physicalFileName(iCatalog, configuredFileName, isPFN));
    }
    return pfns;
  }

  std::string InputFileCatalog::physicalFileName(unsigned int iCatalog,
                                                 std::string const& configuredFileName,
                                                 bool isPFN) const {
    std::string pfn;

    if (isPFN) {
      pfn = configuredFileName;
    } else {
      pfn = pfnFromCatalog(iCatalog, configuredFileName);
    }

    // Append CGI parameters (sciTags) to complete the PFN
    if (sciTagCategory_ != SciTagCategory::Undefined) {
      Service<StorageURLModifier> sciTagConfigService;
      sciTagConfigService->modify(sciTagCategory_, pfn);
    }

    return pfn;
  }

  std::string InputFileCatalog::pfnFromCatalog(unsigned int iCatalog, std::string const& lfn) const {
    std::string pfn;
    if (overrideFileLocator_) {
      pfn = overrideFileLocator_->pfn(lfn);
    } else {
      pfn = fileLocators_[iCatalog]->pfn(lfn);
      if (pfn.empty() && useLFNasPFNifLFNnotFound_) {
        pfn = lfn;
      }
    }
    return pfn;
  }

}  // namespace edm
