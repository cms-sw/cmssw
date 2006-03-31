//////////////////////////////////////////////////////////////////////
//
// $Id: PoolCatalog.cc,v 1.6 2006/03/10 23:24:54 wmtan Exp $
//
// Author: Luca Lista
// Co-Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "IOPool/Common/interface/PoolCatalog.h"
#include "POOLCore/POOLContext.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/IFCAction.h"
#include "FileCatalog/IFCContainer.h"

#include <fstream>

namespace edm {
  PoolCatalog::PoolCatalog(unsigned int rw, std::string url) : catalog_() {
    bool read = rw & READ;
    bool write = rw & WRITE;
    assert(read || write);
    pool::POOLContext::loadComponent("SEAL/Services/MessageService");
    //  POOLContext::setMessageVerbosityLevel(seal::Msg::Info);

    if (url.empty()) {

      if (!write) {
	// For reading use the catalog specified in the site-local
        // config file if that config file exists, otherwise default
        // to file:PoolFileCatalog.xml for now.
        if (0 == getenv("CMS_PATH")) {
          throw cms::Exception("CMSPATHNotFound", 
                               "PoolCatalog::PoolCatalog()\n")
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
              url = buffer;
              std::cout << "CATALOG: " << url << std::endl;
            }
            configFile.close();
          } else {
            // Use the default catalog until the site-local config is
            // really deployed and can be made a required default - PE
            url = "file:PoolFileCatalog.xml";
            //throw cms::Exception("SiteLocalConfigNotFound", 
            //                 "PoolCatalog::PoolCatalog()\n")
            //<< "The site-local data management configuration file was\n";
            //<< "not found. This should be located at:\n";
            //<< "  $CMSPATH/SITECONF/JobConfig/site-local.cfg\n";
          }
        } 
      } else {
        url = "file:PoolFileCatalog.xml"; // always for the write case
      }

    } else {
      url = toPhysical(url);
    }
    pool::URIParser parser(url);
    parser.parse();

    if (read) {
      LogInfo("FwkJob") << "READ_CATALOG: " << parser.contactstring();
      catalog_.addReadCatalog(parser.contactstring());
    }
    if (write) {
      LogInfo("FwkJob") << "WRITE_CATALOG: " << parser.contactstring();
      catalog_.setWriteCatalog(parser.contactstring());
    }
    catalog_.connect();
    catalog_.start();
  }

  PoolCatalog::~PoolCatalog() {
    catalog_.commit();
    catalog_.disconnect();
  }

  void PoolCatalog::commitCatalog() {
    catalog_.commit();
    catalog_.start();
  }

  void PoolCatalog::registerFile(std::string const& pfn, std::string const& lfn) {
    if (!lfn.empty()) {
      pool::FCregister action;
      catalog_.setAction(action);
      action.registerLFN(pfn, lfn);       
    }
  }

  void PoolCatalog::findFile(std::string & pfn, std::string const& lfn) {
    if (isPhysical(lfn)) {
      pfn = lfn;
    } else {
      pool::FClookup action;
      catalog_.setAction(action);
      pool::FileCatalog::FileID fid;
      action.lookupFileByLFN(lfn, fid);
      if (fid == pool::FileCatalog::FileID()) {
        throw cms::Exception("LogicalFileNameNotFound", "PoolCatalog::findFile()\n")
          << "Logical file name " << lfn << " was not found in the file catalog.\n"
	  << "If you wanted a local file, you forgot the 'file:' prefix\n"
	  << "before the file name in your configuration file.\n";
      } else {
        LogInfo("FwkJob") << "LFN: " << lfn;
        std::string fileType;
        action.lookupBestPFN(fid, pool::FileCatalog::READ, pool::FileCatalog::SEQUENTIAL, pfn, fileType);
      }
    }
  }
}
