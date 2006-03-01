#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonCrystalConsistencyDat.h"

using namespace std;
using namespace oracle::occi;

MonCrystalConsistencyDat::MonCrystalConsistencyDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_processedEvents = 0;
  m_problematicEvents = 0;
  m_problemsID = 0;
  m_problemsGainZero = 0;
  m_problemsGainSwitch = 0;
  m_taskStatus = 0;
}



MonCrystalConsistencyDat::~MonCrystalConsistencyDat()
{
}



void MonCrystalConsistencyDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_crystal_consistency_dat (iov_id, logic_id, "
			"processed_events, problematic_events, problems_id, problems_gain_zero, problems_gain_switch, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6, :7, :8)");
  } catch (SQLException &e) {
    throw(runtime_error("MonCrystalConsistencyDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonCrystalConsistencyDat::writeDB(const EcalLogicID* ecid, const MonCrystalConsistencyDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonCrystalConsistencyDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonCrystalConsistencyDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getProcessedEvents() );
    m_writeStmt->setInt(4, item->getProblematicEvents() );
    m_writeStmt->setInt(5, item->getProblemsID() );
    m_writeStmt->setInt(6, item->getProblemsGainZero() );
    m_writeStmt->setInt(7, item->getProblemsGainSwitch() );
    m_writeStmt->setInt(8, item->getTaskStatus() );
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonCrystalConsistencyDat::writeDB():  "+e.getMessage()));
  }
}



void MonCrystalConsistencyDat::fetchData(std::map< EcalLogicID, MonCrystalConsistencyDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonCrystalConsistencyDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.processed_events, d.problematic_events, d.problems_id, d.problems_gain_zero, d.problems_gain_switch, d.task_status "
		 "FROM channelview cv JOIN mon_crystal_consistency_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonCrystalConsistencyDat > p;
    MonCrystalConsistencyDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setProcessedEvents( rset->getInt(7) );
      dat.setProblematicEvents( rset->getInt(8) );
      dat.setProblemsID( rset->getInt(9) );
      dat.setProblemsGainZero( rset->getInt(10) );
      dat.setProblemsGainSwitch( rset->getInt(11) );
      dat.setTaskStatus( rset->getInt(12) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonCrystalConsistencyDat::fetchData():  "+e.getMessage()));
  }
}
