//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDataSvc.cc,v 1.1 2005/11/23 02:17:56 wmtan Exp $
//
// Author: Luca Lista
// Co-Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "IOPool/Common/interface/PoolDataSvc.h"
#include "IOPool/Common/interface/PoolCatalog.h"
#include "DataSvc/IDataSvc.h"
#include "DataSvc/DataSvcContext.h"
#include "DataSvc/DataSvcFactory.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ITechnologySpecificAttributes.h"
#include "StorageSvc/DbLonglong.h"

namespace edm {
  PoolDataSvc::PoolDataSvc(PoolCatalog & catalog_, bool write, bool del) : context_(0) {
    pool::DataSvcContext ctx;
    ctx.setFileCatalog(&catalog_.catalog());
    pool::ObjectDeletePolicy deletePolicy;
    deletePolicy.setOnCache(del);  // delete on the cache
    deletePolicy.setOnRef(del);    // delete on the 'free' ref
    ctx.setObjectDeletePolicy(deletePolicy);
    context_ = pool::DataSvcFactory::create(ctx);
    if (write) {
      pool::DatabaseConnectionPolicy policy;
      policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
      policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::OVERWRITE);
      context_->session().setDefaultConnectionPolicy(policy);
    }
  }

  size_t
  PoolDataSvc::getFileSize(std::string const& fileName) {
    std::auto_ptr<pool::IDatabase>
      idb(context_->session().databaseHandle(fileName, pool::DatabaseSpecification::PFN));
    return idb->technologySpecificAttributes().attribute<pool::DbLonglong>("FILE_SIZE");
  }
}
