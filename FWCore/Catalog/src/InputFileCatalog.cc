//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/Catalog/interface/InputFileCatalog.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/algorithm/string.hpp>

namespace edm {
  InputFileCatalog::InputFileCatalog(edm::ParameterSet const & pset,
				     std::vector<std::string> const& fileNames, bool noThrow) :
    logicalFileNames_(fileNames),
    fileNames_(logicalFileNames_),
    fileCatalogItems_(),
    fl_(pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())){

    // Starting the catalog will write a catalog out if it does not exist.
    // So, do not start (or even read) the catalog unless it is needed.
    
    fileCatalogItems_.reserve(fileNames_.size());
    typedef std::vector<std::string>::iterator iter;
    for(iter it = fileNames_.begin(), lt = logicalFileNames_.begin(), itEnd = fileNames_.end();
	it != itEnd; ++it, ++lt) {
      boost::trim(*it);
      if (it->empty()) {
        throw edm::Exception(edm::errors::Configuration, "InputFileCatalog::InputFileCatalog()\n")
	  << "An empty string specified in the fileNames parameter for input source.\n";
      }
      if (isPhysical(*it)) {
        lt->clear();
      } else {
        boost::trim(*lt);
	findFile(*it, *lt, noThrow);
      }
      fileCatalogItems_.push_back(FileCatalogItem(*it, *lt));
    }
  }
  
  InputFileCatalog::~InputFileCatalog() {}
  
  void InputFileCatalog::findFile(std::string & pfn, std::string const& lfn, bool noThrow) {
    pfn = fl->pfn(lfn);
    if (pfn.empty() && !noThrow) {
      throw cms::Exception("LogicalFileNameNotFound", "FileCatalog::findFile()\n")
	<< "Logical file name '" << lfn << "' was not found in the file catalog.\n"
	<< "If you wanted a local file, you forgot the 'file:' prefix\n"
	<< "before the file name in your configuration file.\n";
    }
  }
  
}
