//////////////////////////////////////////////////////////////////////
//
// $Id: PoolCatalog.cc,v 1.4 2006/01/11 22:33:25 wmtan Exp $
//
// Author: Luca Lista
// Co-Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "IOPool/Common/interface/PoolCatalog.h"
#include "POOLCore/POOLContext.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/IFCAction.h"
#include "FileCatalog/IFCContainer.h"

namespace edm {
  PoolCatalog::PoolCatalog(unsigned int rw, std::string url) : catalog_() {
    bool read = rw & READ;
    bool write = rw & WRITE;
    assert(read || write);
    pool::POOLContext::loadComponent("SEAL/Services/MessageService");
    //  POOLContext::setMessageVerbosityLevel(seal::Msg::Info);

    if (url.empty()) {
       std::string const defaultCatalog = "file:PoolFileCatalog.xml";
       url = defaultCatalog;
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
        LogWarning("FwkJob") << "LFN: " << lfn << " not found";
        pfn = "file:" + lfn;
        LogWarning("FwkJob") << "PFN defaulted: " << pfn ;
      } else {
        LogInfo("FwkJob") << "LFN: " << lfn;
        std::string fileType;
        action.lookupBestPFN(fid, pool::FileCatalog::READ, pool::FileCatalog::SEQUENTIAL, pfn, fileType);
      }
    }
  }
}
