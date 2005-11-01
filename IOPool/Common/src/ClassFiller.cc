#include "IOPool/Common/interface/ClassFiller.h"

#include "StorageSvc/IOODatabaseFactory.h"

namespace edm {
  // ---------------------
  void ClassFiller() {
    pool::IOODatabaseFactory::get()->create(pool::ROOT_StorageType.storageName());
  }
}
