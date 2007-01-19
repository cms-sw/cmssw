//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.cc,v 1.10 2007/01/03 16:07:19 wmtan Exp $
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
#include "StorageSvc/DbType.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/SiteLocalConfig.h"

#include <boost/algorithm/string.hpp>

#include <fstream>

namespace edm {

  FileCatalog::FileCatalog(ParameterSet const& pset) :
      catalog_(),
      url_(pset.getUntrackedParameter<std::string>("catalog", std::string())),
      active_(false) {
    boost::trim(url_);
  }

  FileCatalog::~FileCatalog() {
    if (active_) catalog_.commit();
    catalog_.disconnect();
  }

  void FileCatalog::commitCatalog() {
    catalog_.commit();
    catalog_.start();
  }

  InputFileCatalog::InputFileCatalog(ParameterSet const& pset, bool noThrow) :
    FileCatalog(pset),
    logicalFileNames_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames")),
    fileNames_(logicalFileNames_),
    fileCatalogItems_() {

    if (logicalFileNames_.empty()) {
        throw edm::Exception(edm::errors::Configuration, "InputFileCatalog::InputFileCatalog()\n")
	  << "Empty 'fileNames' parameter specified for input source.\n";
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
	  << "An empty string specified in 'fileNames' parameter for input source.\n";
      }
      if (isPhysical(*it)) {
        lt->clear();
      } else {
        boost::trim(*lt);
	if (!active()) {
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

          catalog().start();
	  setActive();
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

  OutputFileCatalog::OutputFileCatalog(ParameterSet const& pset) :
      FileCatalog(pset),
      fileName_(pset.getUntrackedParameter<std::string>("fileName")),
      logicalFileName_(pset.getUntrackedParameter<std::string>("logicalFileName", std::string())) {
    boost::trim(fileName_);
    if (fileName_.empty()) {
        throw edm::Exception(edm::errors::Configuration, "OutputFileCatalog::OutputFileCatalog()\n")
	  << "Empty 'fileName' parameter specified for output module.\n";
    }
    boost::trim(logicalFileName_);
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

  pool::FileCatalog::FileID OutputFileCatalog::registerFile(std::string const& pfn, std::string const& lfn) {
    pool::FileCatalog::FileID fid;
    {
      std::string type;
      pool::FCregister action;
      catalog().setAction(action);
      action.lookupFileByPFN(pfn, fid, type);
    }
    if (fid.empty()) {
      std::string fileType = "ROOT_Tree";
      pool::FCregister action;
      catalog().setAction(action);
      action.registerPFN(pfn, fileType, fid);
    }
    if (!lfn.empty()) {
      if (isPhysical(lfn)) {
        throw cms::Exception("IllegalCharacter") << "Logical file name '" << lfn
        << "' contains a colon (':'), which is illegal in an LFN.\n";
      }
      pool::FileCatalog::FileID fidl;
      {
        pool::FCregister action;
        catalog().setAction(action);
        action.lookupFileByLFN(lfn, fidl);
      }
      if (fidl.empty()) {
        pool::FCregister action;
        catalog().setAction(action);
        action.registerLFN(pfn, lfn);       
      }
    }
    return fid;
  }
}
