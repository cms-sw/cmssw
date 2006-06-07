//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.cc,v 1.2 2006/05/23 09:13:43 elmer Exp $
//
// Original Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/FileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/IFCAction.h"
#include "FileCatalog/IFCContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/SiteLocalConfig.h"

#include <fstream>

namespace edm {

  FileCatalog::FileCatalog(ParameterSet const& pset) :
      catalog_(),
      url_(pset.getUntrackedParameter<std::string>("catalog", std::string())),
      active_(false) {
  }

  FileCatalog::~FileCatalog() {
    if (active_) catalog_.commit();
    catalog_.disconnect();
  }

  void FileCatalog::commitCatalog() {
    catalog_.commit();
    catalog_.start();
  }

  InputFileCatalog::InputFileCatalog(ParameterSet const& pset) :
    FileCatalog(pset),
    logicalFileNames_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames")),
    fileNames_(logicalFileNames_) {

    if (url().empty()) {
      // For reading use the catalog specified in the site-local config file
      url() = Service<edm::SiteLocalConfig>()->dataCatalog();
      std::cout << "Using the site default catalog: " << url() << std::endl;
    } else {
      url() = toPhysical(url());
    }
    pool::URIParser parser(url());
    parser.parse();

    catalog().addReadCatalog(parser.contactstring());
    catalog().connect();

    // Starting the catalog will write a catalog out if it does not exist.
    // So, do not start the catalog unless it is needed.

    typedef std::vector<std::string>::iterator iter;
    for(iter it = fileNames_.begin(); it != fileNames_.end(); ++it) {
      if (!isPhysical(*it)) {
	if (!active()) {
          catalog().start();
	  setActive();
        }
	findFile(*it, *it);
      }
    }
  }

  InputFileCatalog::~InputFileCatalog() {}

  void InputFileCatalog::findFile(std::string & pfn, std::string const& lfn) {
    pool::FClookup action;
    catalog().setAction(action);
    pool::FileCatalog::FileID fid;
    action.lookupFileByLFN(lfn, fid);
    if (fid == pool::FileCatalog::FileID()) {
      throw cms::Exception("LogicalFileNameNotFound", "FileCatalog::findFile()\n")
        << "Logical file name " << lfn << " was not found in the file catalog.\n"
	<< "If you wanted a local file, you forgot the 'file:' prefix\n"
	<< "before the file name in your configuration file.\n";
    } else {
      std::string fileType;
      action.lookupBestPFN(fid, pool::FileCatalog::READ, pool::FileCatalog::SEQUENTIAL, pfn, fileType);
    }
  }

  OutputFileCatalog::OutputFileCatalog(ParameterSet const& pset) : FileCatalog(pset) {
    if (url().empty()) {
      url() = "file:PoolFileCatalog.xml"; // always for the output case
    } else {
      url() = toPhysical(url());
    }
    pool::URIParser parser(url());
    parser.parse();
    catalog().setWriteCatalog(parser.contactstring());
    catalog().connect();
    catalog().start();
    setActive();
  }

  OutputFileCatalog::~OutputFileCatalog() {}

  void OutputFileCatalog::registerFile(std::string const& pfn, std::string const& lfn) {
    if (!lfn.empty()) {
      pool::FCregister action;
      catalog().setAction(action);
      action.registerLFN(pfn, lfn);       
    }
  }
}
