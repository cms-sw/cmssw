//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Catalog/interface/InputFileCatalog.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <boost/algorithm/string.hpp>

namespace edm {

  InputFileCatalog::InputFileCatalog(std::vector<std::string> const& fileNames, std::string const& override, bool noThrow) :
    logicalFileNames_(fileNames),
    fileNames_(fileNames),
    fallbackFileNames_(fileNames.size()),
    fileCatalogItems_(),
    fileLocator_(),
    fallbackFileLocator_() {

    init(fileNames, override, "", noThrow);
  }

  InputFileCatalog::InputFileCatalog(std::vector<std::string> const& fileNames, std::string const& override, std::string const& overrideFallback, bool noThrow) :
    logicalFileNames_(fileNames),
    fileNames_(fileNames),
    fallbackFileNames_(fileNames.size()),
    fileCatalogItems_(),
    fileLocator_(),
    fallbackFileLocator_() {

    init(fileNames, override, overrideFallback, noThrow);
  }
  
  InputFileCatalog::~InputFileCatalog() {}

  void InputFileCatalog::init(std::vector<std::string> const& fileNames, std::string const& inputOverride, std::string const& inputOverrideFallback, bool noThrow) {

    fileCatalogItems_.reserve(fileNames_.size());
    typedef std::vector<std::string>::iterator iter;
    for(iter it = fileNames_.begin(), lt = logicalFileNames_.begin(), itEnd = fileNames_.end(), ft = fallbackFileNames_.begin();
        it != itEnd; ++it, ++lt, ++ft) {
      boost::trim(*it);
      if (it->empty()) {
        throw Exception(errors::Configuration, "InputFileCatalog::InputFileCatalog()\n")
          << "An empty string specified in the fileNames parameter for input source.\n";
      }
      if (isPhysical(*it)) {
        // Clear the LFN.
        lt->clear();
      } else {
        if (!fileLocator_) {
          fileLocator_.reset(new FileLocator("", false));
        }
        if (!overrideFileLocator_ && !inputOverride.empty()) {
          overrideFileLocator_.reset(new FileLocator(inputOverride, false));
        }
        if (!fallbackFileLocator_) {
          fallbackFileLocator_.reset(new FileLocator("", true));
        }
        if (!overrideFallbackFileLocator_ && !inputOverrideFallback.empty()) {
          overrideFallbackFileLocator_.reset(new FileLocator(inputOverrideFallback, true));
        }
        boost::trim(*lt);
        findFile(*it, *ft, *lt, noThrow);
      }
      fileCatalogItems_.push_back(FileCatalogItem(*it, *lt, *ft));
    }
  }
  
  void InputFileCatalog::findFile(std::string& pfn, std::string& fallbackPfn, std::string const& lfn, bool noThrow) {
    if (overrideFileLocator_) {
      pfn = overrideFileLocator_->pfn(lfn);
    }
    if (pfn.empty())
      pfn = fileLocator_->pfn(lfn);
    if (pfn.empty() && !noThrow) {
      throw cms::Exception("LogicalFileNameNotFound", "FileCatalog::findFile()\n")
	<< "Logical file name '" << lfn << "' was not found in the file catalog.\n"
	<< "If you wanted a local file, you forgot the 'file:' prefix\n"
	<< "before the file name in your configuration file.\n";
    }
    if (overrideFallbackFileLocator_) {
      fallbackPfn = overrideFallbackFileLocator_->pfn(lfn);
    }
    if (fallbackPfn.empty() && fallbackFileLocator_) {
      fallbackPfn = fallbackFileLocator_->pfn(lfn);
      // Empty fallback PFN is OK.
    }
  }
}
