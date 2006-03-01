#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonLaserIRedDat.h"

using namespace std;
using namespace oracle::occi;

MonLaserIRedDat::MonLaserIRedDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_apdMean = 0;
  m_apdRMS = 0;
  m_apdOverPNMean = 0;
  m_apdOverPNRMS = 0;
  m_taskStatus = 0;
  
}



MonLaserIRedDat::~MonLaserIRedDat()
{
}



void MonLaserIRedDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_laser_ired_dat (iov_id, logic_id, "
			"apd_mean, apd_rms, apd_over_pn_mean, apd_over_pn_rms, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":apd_mean, :apd_rms, :apd_over_pn_mean, :apd_over_pn_rms, :task_status)");
  } catch (SQLException &e) {
    throw(runtime_error("MonLaserIRedDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonLaserIRedDat::writeDB(const EcalLogicID* ecid, const MonLaserIRedDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonLaserIRedDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonLaserIRedDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getAPDMean() );
    m_writeStmt->setFloat(4, item->getAPDRMS() );
    m_writeStmt->setFloat(5, item->getAPDOverPNMean() );
    m_writeStmt->setFloat(6, item->getAPDOverPNRMS() );
    m_writeStmt->setInt(7, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonLaserIRedDat::writeDB():  "+e.getMessage()));
  }
}



void MonLaserIRedDat::fetchData(std::map< EcalLogicID, MonLaserIRedDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();

  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonLaserIRedDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.apd_mean, d.apd_rms, d.apd_over_pn_mean, d.apd_over_pn_rms, d.task_status "
		 "FROM channelview cv JOIN mon_laser_ired_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonLaserIRedDat > p;
    MonLaserIRedDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setAPDMean( rset->getFloat(7) );
      dat.setAPDRMS( rset->getFloat(8) );
      dat.setAPDOverPNMean( rset->getFloat(9) );
      dat.setAPDOverPNRMS( rset->getFloat(10) );
      dat.setTaskStatus( rset->getInt(11) );
			

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonLaserIRedDat::fetchData():  "+e.getMessage()));
  }
}
