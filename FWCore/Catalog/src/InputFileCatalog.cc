//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Catalog/interface/InputFileCatalog.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

        edm::CatalogAttributes inputOverride_struct(tmps[0],   //current-site
                                                    tmps[1],   //current-subSite
                                                    tmps[2],   //desired-data-access-site
                                                    tmps[3],   //desired-data-access-volume
                                                    tmps[4]);  //desired-data-access-protocol

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
      //Construct all file locators from data catalogs. If a data catalog is invalid (wrong protocol for example), it is skipped and no file locator is constructed (an exception is thrown out from FileLocator::init).
      for (const auto& catalog : tmp_dataCatalogs) {
        try {
          fileLocators_trivalCatalog_.push_back(std::make_unique<FileLocator>(catalog));
        } catch (cms::Exception const& e) {
          edm::LogWarning("InputFileCatalog")
              << "Caught an exception while constructing a file locator in InputFileCatalog::init: " << e.what()
              << "Skip this catalog";
        }
      }
      if (fileLocators_trivalCatalog_.empty()) {
        cms::Exception ex("FileCatalog");
        ex << "Unable to construct any file locator in InputFileCatalog::init";
        ex.addContext("Calling edm::InputFileCatalog::init()");
        throw ex;
      }
    } else if (catType == edm::CatalogType::RucioCatalog) {
      std::vector<edm::CatalogAttributes> const& tmp_dataCatalogs = localconfservice->dataCatalogs();
      if (!fileLocators_.empty())
        fileLocators_.clear();
      //Construct all file locators from data catalogs. If a data catalog is invalid (wrong protocol for example), it is skipped and no file locator is constructed (an exception is thrown out from FileLocator::init).
      for (const auto& catalog : tmp_dataCatalogs) {
        try {
          fileLocators_.push_back(std::make_unique<FileLocator>(catalog));
        } catch (cms::Exception const& e) {
          edm::LogWarning("InputFileCatalog")
              << "Caught an exception while constructing a file locator in InputFileCatalog::init: " << e.what()
              << "Skip this catalog";
        }
      }
      if (fileLocators_.empty()) {
        cms::Exception ex("FileCatalog");
        ex << "Unable to construct any file locator in InputFileCatalog::init";
        ex.addContext("Calling edm::InputFileCatalog::init()");
        throw ex;
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
