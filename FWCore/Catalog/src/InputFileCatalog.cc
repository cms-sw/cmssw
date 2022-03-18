//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Catalog/interface/InputFileCatalog.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <boost/algorithm/string.hpp>

#include <iostream>

namespace edm {

  InputFileCatalog::InputFileCatalog(std::vector<std::string> const& fileNames,
                                     std::string const& override,
                                     bool useLFNasPFNifLFNnotFound,
                                     edm::CatalogType catType)
      : logicalFileNames_(fileNames), fileNames_(fileNames), fileCatalogItems_(), overrideFileLocator_() {
    init(override, useLFNasPFNifLFNnotFound, catType);
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

  void InputFileCatalog::init(std::string const& inputOverride,
                              bool useLFNasPFNifLFNnotFound,
                              edm::CatalogType catType) {
    typedef std::vector<std::string>::iterator iter;

    if (!overrideFileLocator_ && !inputOverride.empty()) {
      if (catType == edm::CatalogType::TrivialCatalog) {
        overrideFileLocator_ =
            std::make_unique<FileLocator>(inputOverride);  // propagate_const<T> has no reset() function
      } else if (catType == edm::CatalogType::RucioCatalog) {
        //now make a struct from input string
        std::vector<std::string> tmps;
        boost::algorithm::split(tmps, inputOverride, boost::is_any_of(std::string(",")));
        if (tmps.size() != 5) {
          cms::Exception ex("FileCatalog");
          ex << "Trying to override input file catalog but invalid input override string " << inputOverride
             << " (Should be site,subSite,storageSite,volume,protocol)";
          ex.addContext("Calling edm::InputFileCatalog::init()");
          throw ex;
        }

        edm::CatalogAttributes inputOverride_struct(
            tmps[0], tmps[1], tmps[2], tmps[3], tmps[4]);  //site, subSite,storageSite,volume,protocol

        overrideFileLocator_ =
            std::make_unique<FileLocator>(inputOverride_struct);  // propagate_const<T> has no reset() function
      }
    }

    Service<SiteLocalConfig> localconfservice;
    if (!localconfservice.isAvailable()) {
      cms::Exception ex("FileCatalog");
      ex << "edm::SiteLocalConfigService is not available";
      ex.addContext("Calling edm::InputFileCatalog::init()");
      throw ex;
    }

    if (catType == edm::CatalogType::TrivialCatalog) {
      std::vector<std::string> const& tmp_dataCatalogs = localconfservice->trivialDataCatalogs();
      if (!fileLocators_trivalCatalog_.empty())
        fileLocators_trivalCatalog_.clear();

      //require the first file locator to success so obvious mistakes in data catalogs, typos for example, can be catched early. Note that tmp_dataCatalogs is not empty at this point. The protection is done inside the trivialDataCatalogs() of SiteLocalConfigService
      fileLocators_trivalCatalog_.push_back(std::make_unique<FileLocator>(tmp_dataCatalogs.front()));

      for (auto it = tmp_dataCatalogs.begin() + 1; it != tmp_dataCatalogs.end(); ++it) {
        try {
          fileLocators_trivalCatalog_.push_back(std::make_unique<FileLocator>(*it));
        } catch (cms::Exception const& e) {
          continue;
        }
      }
    } else if (catType == edm::CatalogType::RucioCatalog) {
      std::vector<edm::CatalogAttributes> const& tmp_dataCatalogs = localconfservice->dataCatalogs();
      if (!fileLocators_.empty())
        fileLocators_.clear();
      //require the first file locator to success so obvious mistakes in data catalogs, typos for example, can be catched early. Note that tmp_dataCatalogs is not empty at this point. The protection is done inside the dataCatalogs() of SiteLocalConfigService
      fileLocators_.push_back(std::make_unique<FileLocator>(tmp_dataCatalogs.front()));
      for (auto it = tmp_dataCatalogs.begin() + 1; it != tmp_dataCatalogs.end(); ++it) {
        try {
          fileLocators_.push_back(std::make_unique<FileLocator>(*it));
        } catch (cms::Exception const& e) {
          continue;
        }
      }
    } else {
      cms::Exception ex("FileCatalog");
      ex << "Undefined catalog type";
      ex.addContext("Calling edm::InputFileCatalog::init()");
      throw ex;
    }

    for (iter it = fileNames_.begin(), lt = logicalFileNames_.begin(), itEnd = fileNames_.end(); it != itEnd;
         ++it, ++lt) {
      boost::trim(*it);
      std::vector<std::string> pfns;
      if (it->empty()) {
        cms::Exception ex("FileCatalog");
        ex << "An empty string specified in the fileNames parameter for input source";
        ex.addContext("Calling edm::InputFileCatalog::init()");
        throw ex;
      }
      if (isPhysical(*it)) {
        if (it->back() == ':') {
          cms::Exception ex("FileCatalog");
          ex << "An empty physical file name specified in the fileNames parameter for input source";
          ex.addContext("Calling edm::InputFileCatalog::init()");
          throw ex;
        }
        pfns.push_back(*it);
        // Clear the LFN.
        lt->clear();
      } else {
        boost::trim(*lt);
        findFile(*lt, pfns, useLFNasPFNifLFNnotFound, catType);
      }

      fileCatalogItems_.push_back(FileCatalogItem(pfns, *lt));
    }
  }

  void InputFileCatalog::findFile(std::string const& lfn,
                                  std::vector<std::string>& pfns,
                                  bool useLFNasPFNifLFNnotFound,
                                  edm::CatalogType catType) {
    if (overrideFileLocator_) {
      pfns.push_back(overrideFileLocator_->pfn(lfn, catType));
    } else {
      if (catType == edm::CatalogType::TrivialCatalog) {
        for (auto const& locator : fileLocators_trivalCatalog_) {
          std::string pfn = locator->pfn(lfn, edm::CatalogType::TrivialCatalog);
          if (pfn.empty() && useLFNasPFNifLFNnotFound)
            pfns.push_back(lfn);
          else
            pfns.push_back(pfn);
        }
      } else if (catType == edm::CatalogType::RucioCatalog) {
        for (auto const& locator : fileLocators_) {
          std::string pfn = locator->pfn(lfn, edm::CatalogType::RucioCatalog);
          if (pfn.empty() && useLFNasPFNifLFNnotFound)
            pfns.push_back(lfn);
          else
            pfns.push_back(pfn);
        }
      } else {
        cms::Exception ex("FileCatalog");
        ex << "Undefined catalog type";
        ex.addContext("Calling edm::InputFileCatalog::findFile()");
        throw ex;
      }
    }

    // Empty PFN will be found by caller.
  }

}  // namespace edm
