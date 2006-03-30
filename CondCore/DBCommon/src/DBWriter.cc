#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/Placement.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingUtilities.h"
cond::DBWriter::DBWriter( cond::DBSession& session,
			  const std::string& containerName)
  :m_session(session),m_containerName(containerName),m_placement(new pool::Placement)
{
  m_placement->setTechnology(pool::POOL_RDBMS_StorageType.type());
  m_placement->setDatabase(session.connectionString(), pool::DatabaseSpecification::PFN);
  m_placement->setContainerName(m_containerName);
}
cond::DBWriter::DBWriter( cond::DBSession& session,
			  const std::string& containerName,
			  const std::string& mappingFileName):
  m_session(session),m_containerName(containerName),m_placement(new pool::Placement),m_mappinginput(mappingFileName)
{
  m_placement->setTechnology(pool::POOL_RDBMS_StorageType.type());
  m_placement->setDatabase(session.connectionString(), pool::DatabaseSpecification::PFN);
  m_placement->setContainerName(m_containerName);
  pool::ObjectRelationalMappingUtilities utility;
  std::string connectionString = m_session.connectionString();
  utility.setProperty(pool::ObjectRelationalMappingUtilities::connectionStringProperty(),connectionString);
  utility.buildAndStoreMapping(m_mappinginput);
}
cond::DBWriter::~DBWriter(){
  delete m_placement;
}


