//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <boost/algorithm/string.hpp>

namespace edm {
  InputFileCatalog::InputFileCatalog(ParameterSet const& pset,
				     std::string const& namesParameter,
				     bool canBeEmpty, bool noThrow) :
    logicalFileNames_(canBeEmpty ?
		      pset.getUntrackedParameter<std::vector<std::string> >(namesParameter, std::vector<std::string>()) :
		      pset.getUntrackedParameter<std::vector<std::string> >(namesParameter)),
    fileNames_(logicalFileNames_),
    fileCatalogItems_() {

    if (logicalFileNames_.empty()) {
      if (canBeEmpty) return;
      throw edm::Exception(edm::errors::Configuration, "InputFileCatalog::InputFileCatalog()\n")
	<< "Empty '" << namesParameter << "' parameter specified for input source.\n";
    }
    // Starting the catalog will write a catalog out if it does not exist.
    // So, do not start (or even read) the catalog unless it is needed.
    
    fileCatalogItems_.reserve(fileNames_.size());
    typedef std::vector<std::string>::iterator iter;
    for(iter it = fileNames_.begin(), lt = logicalFileNames_.begin(), itEnd = fileNames_.end();
	it != itEnd; ++it, ++lt) {
      boost::trim(*it);
      if (it->empty()) {
        throw edm::Exception(edm::errors::Configuration, "InputFileCatalog::InputFileCatalog()\n")
	  << "An empty string specified in '" << namesParameter << "' parameter for input source.\n";
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
    pfn = fileLocator().pfn(lfn);
    if (pfn.empty() && !noThrow) {
      throw cms::Exception("LogicalFileNameNotFound", "FileCatalog::findFile()\n")
	<< "Logical file name '" << lfn << "' was not found in the file catalog.\n"
	<< "If you wanted a local file, you forgot the 'file:' prefix\n"
	<< "before the file name in your configuration file.\n";
    }
  }
  
  void
  InputFileCatalog::fillDescription(ParameterSetDescription & desc) {
  }
  
}
