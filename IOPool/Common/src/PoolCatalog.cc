//////////////////////////////////////////////////////////////////////
//
// $Id: PoolCatalog.cc,v 1.4 2005/07/27 12:37:32 wmtan Exp $
//
// Author: Luca Lista
//
//////////////////////////////////////////////////////////////////////

#include "IOPool/Common/interface/PoolCatalog.h"
#include "POOLCore/POOLContext.h"
#include "FileCatalog/URIParser.h"
#include "DataSvc/DataSvcFactory.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "DataSvc/IDataSvc.h"
#include "DataSvc/DataSvcContext.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"
#include "FileCatalog/IFCAction.h"
#include "FileCatalog/IFCContainer.h"

using namespace std;
using namespace pool;

namespace edm {
  PoolCatalog::PoolCatalog(unsigned int rw, std::string const& url) : catalog_() {
    bool read = rw & READ;
    bool write = rw & WRITE;
    assert(read || write);
    POOLContext::loadComponent("SEAL/Services/MessageService");
    //  POOLContext::setMessageVerbosityLevel(seal::Msg::Info);

    URIParser parser(url);
    parser.parse();

    if (read)
      catalog_.addReadCatalog(parser.contactstring());
    if (write)
      catalog_.setWriteCatalog(parser.contactstring());
    catalog_.connect();
    catalog_.start();
  }

  PoolCatalog::~PoolCatalog() {
    //  _context->session().disconnectAll();
    catalog_.commit();
    catalog_.disconnect();
  }

  IDataSvc * PoolCatalog::createContext(bool write, bool del) {
    pool::DataSvcContext ctx;
    ctx.setFileCatalog(&catalog_);
    ObjectDeletePolicy deletePolicy;
    deletePolicy.setOnCache(del);  // delete on the cache
    deletePolicy.setOnRef(del);    // delete on the 'free' ref
    ctx.setObjectDeletePolicy(deletePolicy);
    IDataSvc * cache = pool::DataSvcFactory::create(ctx);
    if (write) {
      pool::DatabaseConnectionPolicy policy;
      policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
      policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::OVERWRITE);
      cache->session().setDefaultConnectionPolicy(policy);
    }
    return cache;
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
