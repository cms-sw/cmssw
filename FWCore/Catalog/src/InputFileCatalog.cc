//////////////////////////////////////////////////////////////////////
//
//
// Original Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/IFCAction.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <boost/algorithm/string.hpp>

namespace edm {
  InputFileCatalog::InputFileCatalog(ParameterSet const& pset,
				     PoolCatalog & poolcat,
				     std::string const& namesParameter,
				     bool canBeEmpty, bool noThrow) :
    FileCatalog(poolcat),
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
	if (!active()) {
	  // Protect against handling the catalog twice.
	  // There is only one catalog, so there is no need
	  // to support multiple different catalogs.
          if (catalog().nReadCatalogs() == 0) {
	    // Add the override catalog, if specified in the pset.

            // The default value provided as the second argument to the getUntrackedParameter function call
            // is not used when the ParameterSet has been validated and the parameters are not optional
            // in the description.  As soon as all primary input sources and all modules with a secondary
            // input sources have defined descriptions, the defaults in the getUntrackedParameterSet function
	    // calls can and should be deleted from the code.
	    std::string overriderUrl = pset.getUntrackedParameter<std::string>("overrideCatalog", std::string());
            if (!overriderUrl.empty ()) {
              pool::URIParser overriderParser(overriderUrl);
              overriderParser.parse();
              catalog().addReadCatalog(overriderParser.contactstring());
	    }
	    // For reading use the catalog specified in the site-local config file
	    url() = Service<edm::SiteLocalConfig>()->dataCatalog();
	    pool::URIParser parser(url());
	    parser.parse();

            catalog().addReadCatalog(parser.contactstring());
            catalog().connect();

            catalog().start();
	    setActive();
	  }
        }
	findFile(*it, *lt, noThrow);
      }
      fileCatalogItems_.push_back(FileCatalogItem(*it, *lt));
    }
  }

  InputFileCatalog::~InputFileCatalog() {}

  void InputFileCatalog::findFile(std::string & pfn, std::string const& lfn, bool noThrow) {
    pool::FClookup action;
    catalog().setAction(action);
    pool::FileCatalog::FileID fid;
    action.lookupFileByLFN(lfn, fid);
    if (fid == pool::FileCatalog::FileID()) {
      if (noThrow) {
	pfn.clear();
      } else {
        throw cms::Exception("LogicalFileNameNotFound", "FileCatalog::findFile()\n")
	  << "Logical file name '" << lfn << "' was not found in the file catalog.\n"
	  << "If you wanted a local file, you forgot the 'file:' prefix\n"
	  << "before the file name in your configuration file.\n";
      }
    } else {
      std::string fileType;
      action.lookupBestPFN(fid, pool::FileCatalog::READ, pool::FileCatalog::SEQUENTIAL, pfn, fileType);
      if (pfn.empty() && !noThrow) {
        throw cms::Exception("LogicalFileNameNotFound", "FileCatalog::findFile()\n")
	  << "Logical file name '" << lfn << "' was not found in the file catalog.\n"
	  << "If you wanted a local file, you forgot the 'file:' prefix\n"
	  << "before the file name in your configuration file.\n";
      }
    }
  }

  void
  InputFileCatalog::fillDescription(ParameterSetDescription & desc) {
    std::string defaultString;
    desc.addUntracked<std::string>("overrideCatalog", defaultString);
  }
}
