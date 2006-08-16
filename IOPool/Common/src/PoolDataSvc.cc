//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDataSvc.cc,v 1.3 2006/04/06 23:45:51 wmtan Exp $
//
// Author: Luca Lista
// Co-Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "IOPool/Common/interface/PoolDataSvc.h"
#include "FWCore/Framework/interface/FileCatalog.h"
#include "StorageSvc/DbLonglong.h"
#include "PersistencySvc/IDatabase.h"
#include "DataSvc/IDataSvc.h"
#include "PersistencySvc/ITechnologySpecificAttributes.h"
#include "PersistencySvc/ISession.h"
#include "DataSvc/DataSvcContext.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "DataSvc/DataSvcFactory.h"

namespace edm {
  PoolDataSvc::PoolDataSvc(InputFileCatalog & catalog_, bool del) : context_(0) {
    pool::DataSvcContext ctx;
    ctx.setFileCatalog(&catalog_.catalog());
    pool::ObjectDeletePolicy deletePolicy;
    deletePolicy.setOnCache(del);  // delete on the cache
    deletePolicy.setOnRef(del);    // delete on the 'free' ref
    ctx.setObjectDeletePolicy(deletePolicy);
    context_ = pool::DataSvcFactory::create(ctx);
  }

  PoolDataSvc::PoolDataSvc(OutputFileCatalog & catalog_, bool del) : context_(0) {
    pool::DataSvcContext ctx;
    ctx.setFileCatalog(&catalog_.catalog());
    pool::ObjectDeletePolicy deletePolicy;
    deletePolicy.setOnCache(del);  // delete on the cache
    deletePolicy.setOnRef(del);    // delete on the 'free' ref
    ctx.setObjectDeletePolicy(deletePolicy);
    context_ = pool::DataSvcFactory::create(ctx);

    pool::DatabaseConnectionPolicy policy;
    policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
    policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::OVERWRITE);
    context_->session().setDefaultConnectionPolicy(policy);
  }

  size_t
  PoolDataSvc::getFileSize(std::string const& fileName) const {
    return getAttribute<pool::DbLonglong>(std::string("FILE_SIZE"), fileName);
  }

  void
  PoolDataSvc::setCompressionLevel(std::string const& fileName, int value) const {
      setAttribute<int>(std::string("COMPRESSION_LEVEL"), fileName, value);
  }

  // These templated functions are called only from this file, so they need not be in a header.
  template <typename T>
  T
  PoolDataSvc::getAttribute(std::string const& attributeName, std::string const& fileName) const {
    std::auto_ptr<pool::IDatabase>
      idb(context_->session().databaseHandle(fileName, pool::DatabaseSpecification::PFN));
    return idb->technologySpecificAttributes().template attribute<T>(attributeName);
  }

  template <typename T>
  void
  PoolDataSvc::setAttribute(std::string const& attributeName, std::string const& fileName, T const& value) const {
    std::auto_ptr<pool::IDatabase>
      idb(context_->session().databaseHandle(fileName, pool::DatabaseSpecification::PFN));
      idb->technologySpecificAttributes().template setAttribute<T>(attributeName, value);
  }

}
