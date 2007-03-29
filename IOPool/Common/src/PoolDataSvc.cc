//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDataSvc.cc,v 1.6 2007/03/04 06:22:37 wmtan Exp $
//
// Author: Luca Lista
// Co-Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "IOPool/Common/interface/PoolDataSvc.h"
#include "FWCore/Catalog/interface/FileCatalog.h"
#include "DataSvc/IDataSvc.h"
#include "PersistencySvc/ISession.h"
#include "DataSvc/DataSvcContext.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "DataSvc/DataSvcFactory.h"

namespace edm {
  PoolDataSvc::PoolDataSvc(InputFileCatalog & catalog_, bool del) : context_() {
    pool::DataSvcContext ctx;
    ctx.setFileCatalog(&catalog_.catalog());
    pool::ObjectDeletePolicy deletePolicy;
    deletePolicy.setOnCache(del);  // delete on the cache
    deletePolicy.setOnRef(del);    // delete on the 'free' ref
    ctx.setObjectDeletePolicy(deletePolicy);
    context_ = boost::shared_ptr<pool::IDataSvc>(pool::DataSvcFactory::create(ctx));
  }

  PoolDataSvc::PoolDataSvc(OutputFileCatalog & catalog_, bool del) : context_() {
    pool::DataSvcContext ctx;
    ctx.setFileCatalog(&catalog_.catalog());
    pool::ObjectDeletePolicy deletePolicy;
    deletePolicy.setOnCache(del);  // delete on the cache
    deletePolicy.setOnRef(del);    // delete on the 'free' ref
    ctx.setObjectDeletePolicy(deletePolicy);
    context_ = boost::shared_ptr<pool::IDataSvc>(pool::DataSvcFactory::create(ctx));

    pool::DatabaseConnectionPolicy policy;
    policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
    policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::OVERWRITE);
    context_->session().setDefaultConnectionPolicy(policy);
  }
}
