//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDatabase.cc,v 1.6 2007/03/04 06:22:37 wmtan Exp $
//
// Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "IOPool/Common/interface/PoolDatabase.h"
#include "IOPool/Common/interface/PoolDataSvc.h"
#include "StorageSvc/DbLonglong.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/IDatabase.h"
#include "DataSvc/IDataSvc.h"
#include "PersistencySvc/ITechnologySpecificAttributes.h"
#include "PersistencySvc/ISession.h"
#include "DataSvc/DataSvcContext.h"

namespace edm {
  PoolDatabase::PoolDatabase(std::string const& fileName, PoolDataSvc const& dataSvc) :
      fileName_(fileName),
      databaseHandle_(
        boost::shared_ptr<pool::IDatabase>(
          dataSvc.context()->session().databaseHandle(fileName_, pool::DatabaseSpecification::PFN))) {
    databaseHandle_->setTechnology(pool::ROOTTREE_StorageType.type());
    databaseHandle_->connectForWrite();
  }

  size_t
  PoolDatabase::getFileSize() const {
    return getAttribute<pool::DbLonglong>(std::string("FILE_SIZE"));
  }

  void
  PoolDatabase::setCompressionLevel(int value) const {
      setAttribute<int>(std::string("COMPRESSION_LEVEL"), value);
  }

  void
  PoolDatabase::setBasketSize(int value) const {
      setAttribute<int>(std::string("DEFAULT_BUFFERSIZE"), value);
  }

  void
  PoolDatabase::setSplitLevel(int value) const {
      setAttribute<int>(std::string("DEFAULT_SPLITLEVEL"), value);
  }

  // These templated functions are called only from this file, so they need not be in a header.
  template <typename T>
  T
  PoolDatabase::getAttribute(std::string const& attributeName) const {
    return databaseHandle_->technologySpecificAttributes().template attribute<T>(attributeName);
  }

  template <typename T>
  void
  PoolDatabase::setAttribute(std::string const& attributeName, T const& value) const {
      databaseHandle_->technologySpecificAttributes().template setAttribute<T>(attributeName, value);
  }

}
