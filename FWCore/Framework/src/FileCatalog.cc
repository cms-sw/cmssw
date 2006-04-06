//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.cc,v 1.7 2006/03/31 18:53:05 elmer Exp $
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

#include <fstream>

namespace edm {

  FileCatalog::FileCatalog(ParameterSet const& pset) :
      catalog_(),
      url_(pset.getUntrackedParameter<std::string>("catalog", std::string())) {
  }

  FileCatalog::~FileCatalog() {
    catalog_.commit();
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
      // For reading use the catalog specified in the site-local
      // config file if that config file exists, otherwise default
      // to file:PoolFileCatalog.xml for now.
      if (0 == getenv("CMS_PATH")) {
        throw cms::Exception("CMSPATHNotFound", 
                             "FileCatalog::FileCatalog()\n")
        << "CMS_PATH envvar is not set, this is required to find the \n"
        << "site-local data management configuration. \n";
      }
      std::string const configDir = getenv("CMS_PATH");
      if (!configDir.empty()) {
        std::string const configFileName = configDir 
                               + "/SITECONF/JobConfig/site-local.cfg";
        std::ifstream configFile(configFileName.c_str());
        if (configFile) {
          char buffer[1024];
          configFile.get(buffer, 1024);
          if (configFile) {
            url() = buffer;
            std::cout << "CATALOG: " << url() << std::endl;
          }
          configFile.close();
        } else {
          // Use the default catalog until the site-local config is
          // really deployed and can be made a required default - PE
          url() = "file:PoolFileCatalog.xml";
          //throw cms::Exception("SiteLocalConfigNotFound", 
          //                 "FileCatalog::FileCatalog()\n")
          //<< "The site-local data management configuration file was\n";
          //<< "not found. This should be located at:\n";
          //<< "  $CMSPATH/SITECONF/JobConfig/site-local.cfg\n";
        }
      } 
    } else {
      url() = toPhysical(url());
    }
    pool::URIParser parser(url());
    parser.parse();

    catalog().addReadCatalog(parser.contactstring());
    catalog().connect();
    catalog().start();

    typedef std::vector<std::string>::iterator iter;
    for(iter it = fileNames_.begin(); it != fileNames_.end(); ++it) {
      if (!isPhysical(*it)) {
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
