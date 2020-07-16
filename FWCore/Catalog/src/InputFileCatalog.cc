//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Catalog/interface/InputFileCatalog.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <boost/algorithm/string.hpp>

namespace edm {

  InputFileCatalog::InputFileCatalog(std::vector<std::string> const& fileNames,
                                     std::string const& override,
                                     bool useLFNasPFNifLFNnotFound)
      : logicalFileNames_(fileNames), fileNames_(fileNames), fileCatalogItems_(), overrideFileLocator_() {
    init(override, useLFNasPFNifLFNnotFound);
  }

  InputFileCatalog::~InputFileCatalog() {}

  std::vector<std::string> InputFileCatalog::fileNames(unsigned iCatalog) const {
    std::vector<std::string> tmp;
    tmp.reserve(fileCatalogItems_.size());
    for (auto const& item : fileCatalogItems_) {
      tmp.push_back(item.fileName(iCatalog));
    }
    return tmp;
  }

  void InputFileCatalog::init(std::string const& inputOverride, bool useLFNasPFNifLFNnotFound) {
    typedef std::vector<std::string>::iterator iter;

    if (!overrideFileLocator_ && !inputOverride.empty()) {
      overrideFileLocator_ =
          std::make_unique<FileLocator>(inputOverride);  // propagate_const<T> has no reset() function
    }

    Service<SiteLocalConfig> localconfservice;
    if (!localconfservice.isAvailable())
      throw cms::Exception("TrivialFileCatalog", "edm::SiteLocalConfigService is not available");

    std::vector<std::string> const& tmp_dataCatalogs = localconfservice->dataCatalogs();
    if (!fileLocators_.empty())
      fileLocators_.clear();

    //require the first file locator to success so obvious mistakes in data catalogs, typos for example, can be catched early. Note that tmp_dataCatalogs is not empty at this point. The protection is done inside the dataCatalogs() above
    fileLocators_.push_back(std::make_unique<FileLocator>(tmp_dataCatalogs.front()));

    for (auto it = tmp_dataCatalogs.begin() + 1; it != tmp_dataCatalogs.end(); ++it) {
      try {
        fileLocators_.push_back(std::make_unique<FileLocator>(*it));
      } catch (cms::Exception const& e) {
        continue;
      }
    }

    for (iter it = fileNames_.begin(), lt = logicalFileNames_.begin(), itEnd = fileNames_.end(); it != itEnd;
         ++it, ++lt) {
      boost::trim(*it);
      std::vector<std::string> pfns;
      if (it->empty()) {
        throw Exception(errors::Configuration, "InputFileCatalog::InputFileCatalog()\n")
            << "An empty string specified in the fileNames parameter for input source.\n";
      }
      if (isPhysical(*it)) {
        if (it->back() == ':') {
          throw Exception(errors::Configuration, "InputFileCatalog::InputFileCatalog()\n")
              << "An empty physical file name specified in the fileNames parameter for input source.\n";
        }
        pfns.push_back(*it);
        // Clear the LFN.
        lt->clear();
      } else {
        boost::trim(*lt);
        findFile(*lt, pfns, useLFNasPFNifLFNnotFound);
      }
      fileCatalogItems_.push_back(FileCatalogItem(pfns, *lt));
    }
  }

  void InputFileCatalog::findFile(std::string const& lfn,
                                  std::vector<std::string>& pfns,
                                  bool useLFNasPFNifLFNnotFound) {
    if (overrideFileLocator_) {
      pfns.push_back(overrideFileLocator_->pfn(lfn));
    } else {
      for (auto const& locator : fileLocators_) {
        std::string pfn = locator->pfn(lfn);
        if (pfn.empty() && useLFNasPFNifLFNnotFound)
          pfns.push_back(lfn);
        else
          pfns.push_back(pfn);
      }
    }

    // Empty PFN will be found by caller.
  }

}  // namespace edm
