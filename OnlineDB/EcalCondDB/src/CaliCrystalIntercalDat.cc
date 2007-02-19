#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/CaliCrystalIntercalDat.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"

using namespace std;
using namespace oracle::occi;

CaliCrystalIntercalDat::CaliCrystalIntercalDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_cali = 0;
  m_caliRMS = 0;
  m_numEvents = 0;
  m_taskStatus = false;
}



CaliCrystalIntercalDat::~CaliCrystalIntercalDat()
{
}



void CaliCrystalIntercalDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();
  
  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO cali_crystal_intercal_dat (iov_id, logic_id, "
			"cali, cali_rms, num_events, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6)");
  } catch (SQLException &e) {
    throw(runtime_error("CaliCrystalIntercalDat::prepareWrite():  "+e.getMessage()));
  }
}



void CaliCrystalIntercalDat::writeDB(const EcalLogicID* ecid, const CaliCrystalIntercalDat* item, CaliIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();
  
  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("CaliCrystalIntercalDat::writeDB:  IOV not in DB")); }
  
  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("CaliCrystalIntercalDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    
    m_writeStmt->setFloat(3, item->getCali() );
    m_writeStmt->setFloat(4, item->getCaliRMS() );
    m_writeStmt->setInt(5, item->getNumEvents() );
    m_writeStmt->setInt(6, item->getTaskStatus() );
    
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("CaliCrystalIntercalDat::writeDB():  "+e.getMessage()));
  }
}



void CaliCrystalIntercalDat::fetchData(std::map< EcalLogicID, CaliCrystalIntercalDat >* fillMap, CaliIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();
  
  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("CaliCrystalIntercalDat::writeDB:  IOV not in DB")); 
    return;
  }
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.cali, d.cali_rms, d.num_events, d.task_status "
		 "FROM channelview cv JOIN cali_crystal_intercal_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, CaliCrystalIntercalDat > p;
    CaliCrystalIntercalDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to
      
      dat.setCali( rset->getFloat(7) );
      dat.setCaliRMS( rset->getFloat(8) );
      dat.setNumEvents( rset->getInt(9) );
      dat.setTaskStatus( rset->getInt(10) );
      
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("CaliCrystalIntercalDat::fetchData():  "+e.getMessage()));
  }
}
