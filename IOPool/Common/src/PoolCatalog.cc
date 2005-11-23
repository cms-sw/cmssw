//////////////////////////////////////////////////////////////////////
//
// $Id: PoolCatalog.cc,v 1.1 2005/11/01 22:42:45 wmtan Exp $
//
// Author: Luca Lista
// Co-Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "IOPool/Common/interface/PoolCatalog.h"
#include "POOLCore/POOLContext.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/IFCAction.h"
#include "FileCatalog/IFCContainer.h"

namespace edm {
  PoolCatalog::PoolCatalog(unsigned int rw, std::string const& url) : catalog_() {
    bool read = rw & READ;
    bool write = rw & WRITE;
    assert(read || write);
    pool::POOLContext::loadComponent("SEAL/Services/MessageService");
    //  POOLContext::setMessageVerbosityLevel(seal::Msg::Info);

    pool::URIParser parser(url);
    parser.parse();

    if (read)
      catalog_.addReadCatalog(parser.contactstring());
    if (write)
      catalog_.setWriteCatalog(parser.contactstring());
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
      // BEGIN KLUDGE due to pool bug: 
      if (pfn.find("file:") == 0) {
        // file: fails if catalog is not present.
        pool::FClookup action;
        catalog_.setAction(action);
        pool::FileCatalog::FileID fid;
        std::string fileType;
        action.lookupFileByPFN(pfn, fid, fileType);
        if (fid == pool::FileCatalog::FileID()) {
          // strip off "file:"
          pfn = pfn.substr(5);
        }
      } // END KLUDGE
    } else {
      pool::FClookup action;
      catalog_.setAction(action);
      pool::FileCatalog::FileID fid;
      action.lookupFileByLFN(lfn, fid);
      if (fid == pool::FileCatalog::FileID()) {
        pfn = lfn;
      } else {
        std::string fileType;
        action.lookupBestPFN(fid, pool::FileCatalog::READ, pool::FileCatalog::SEQUENTIAL, pfn, fileType);
      }
    }
  }
}
