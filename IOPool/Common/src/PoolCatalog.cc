//////////////////////////////////////////////////////////////////////
//
// $Id: PoolCatalog.cc,v 1.5 2006/01/20 20:45:14 wmtan Exp $
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
      url = "file:PoolFileCatalog.xml";
/*
      if (!write) {
	// For reading make the trivial file catalog the default.
	std::string const configDir = getenv("CMS_PATH");
	if (!configDir.empty()) {
          std::string const configFileName = configDir + "/site-local.cfg";
	  std::ifstream configFile(configFileName.c_str());
	  if (configFile) {
	    char buffer[1024];
	    configFile.get(buffer, 1024);
	    if (configFile) {
	      url = buffer;
              std::cout << "CATALOG: " << url << std::endl;
	    }
	    configFile.close();
	  }
	}
      }
*/
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
