#include "CondCore/DBCommon/interface/DBWriter.h"
//#include "CondCore/DBCommon/ServiceLoader.h"
#include "CondCore/DBCommon/interface/DBSession.h"
//#include "FileCatalog/URIParser.h"
//#include "FileCatalog/FCSystemTools.h"
//#include "FileCatalog/IFileCatalog.h"
#include "StorageSvc/DbType.h"
//#include "PersistencySvc/DatabaseConnectionPolicy.h"
//#include "PersistencySvc/ISession.h"
//#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/Placement.h"
//#include "DataSvc/DataSvcFactory.h"
//#include "DataSvc/IDataSvc.h"
//#include "SealKernel/Exception.h"
//#include <algorithm>

cond::DBWriter::DBWriter( cond::DBSession& session,
			  const std::string& containerName)
  :m_session(session),m_containerName(containerName),m_placement(new pool::Placement)
{
  m_placement->setTechnology(pool::POOL_RDBMS_StorageType.type());
  m_placement->setDatabase(session.connectionString(), pool::DatabaseSpecification::PFN);
  m_placement->setContainerName(m_containerName);
}
cond::DBWriter::~DBWriter(){
  delete m_placement;
}


