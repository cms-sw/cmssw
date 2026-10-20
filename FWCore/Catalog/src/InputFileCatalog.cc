#include <filesystem>
#include <utility>

#include <boost/algorithm/string.hpp>

#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  std::string const emptyString;
}

namespace edm {

  InputFileCatalog::InputFileCatalog(std::vector<std::string> configuredFileNames,
                                     std::string const& override,
                                     bool useLFNasPFNifLFNnotFound)
      : configuredFileNames_(std::move(configuredFileNames)), useLFNasPFNifLFNnotFound_(useLFNasPFNifLFNnotFound) {
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
      configuredFileName.shrink_to_fit();
    }
  }

  InputFileCatalog::~InputFileCatalog() = default;

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

    bool const isPFN = isPhysicalFileName(configuredFileName);
    unsigned int const numberOfPFNs = (isPFN || overrideFileLocator_) ? 1 : fileLocators_.size();
    pfns.reserve(numberOfPFNs);
    for (unsigned int iCatalog = 0; iCatalog < numberOfPFNs; ++iCatalog) {
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
    std::string const& firstConfiguredFileName = configuredFileNames_.front();
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
    return isPFN ? configuredFileName : pfnFromCatalog(iCatalog, configuredFileName);
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
