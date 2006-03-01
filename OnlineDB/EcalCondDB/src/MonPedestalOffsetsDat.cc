#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonPedestalOffsetsDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

using namespace std;
using namespace oracle::occi;

MonPedestalOffsetsDat::MonPedestalOffsetsDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_dacG1 = 0;
  m_dacG6 = 0;
  m_dacG12 = 0;
  m_taskStatus = 0;
}



MonPedestalOffsetsDat::~MonPedestalOffsetsDat()
{
}



void MonPedestalOffsetsDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_pedestal_offsets_dat (iov_id, logic_id, "
			"dac_g1, dac_g6, dac_g12, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":dac_g1, :dac_g6, :dac_g12, :task_status)");
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalOffsetsDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonPedestalOffsetsDat::writeDB(const EcalLogicID* ecid, const MonPedestalOffsetsDat* item, MonRunIOV* iov )
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonPedestalOffsetsDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonPedestalOffsetsDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getDACG1() );
    m_writeStmt->setInt(4, item->getDACG6() );
    m_writeStmt->setInt(5, item->getDACG12() );
    m_writeStmt->setInt(6, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalOffsetsDat::writeDB():  "+e.getMessage()));
  }
}



void MonPedestalOffsetsDat::fetchData(std::map< EcalLogicID, MonPedestalOffsetsDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonPedestalOffsetsDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.dac_g1, d.dac_g6, d.dac_g12, d.task_status "
		 "FROM channelview cv JOIN mon_pedestal_offsets_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonPedestalOffsetsDat > p;
    MonPedestalOffsetsDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setDACG1( rset->getInt(7) );
      dat.setDACG6( rset->getInt(8) );
      dat.setDACG12( rset->getInt(9) );
      dat.setTaskStatus( rset->getInt(10) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalOffsetsDat::fetchData():  "+e.getMessage()));
  }
}
