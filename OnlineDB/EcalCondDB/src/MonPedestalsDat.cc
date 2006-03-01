#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

using namespace std;
using namespace oracle::occi;

MonPedestalsDat::MonPedestalsDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_pedMeanG1 = 0;
  m_pedMeanG6 = 0;
  m_pedMeanG12 = 0;
  m_pedRMSG1 = 0;
  m_pedRMSG6 = 0;
  m_pedRMSG12 = 0;
  m_taskStatus = 0;
}



MonPedestalsDat::~MonPedestalsDat()
{
}



void MonPedestalsDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_pedestals_dat (iov_id, logic_id, "
		      "ped_mean_g1, ped_mean_g6, ped_mean_g12, "
		      "ped_rms_g1, ped_rms_g6, ped_rms_g12, task_status) "
		      "VALUES (:iov_id, :logic_id, "
		      ":ped_mean_g1, :ped_mean_g6, :ped_mean_g12, "
		      ":ped_rms_g1, :ped_rms_g6, :ped_rms_g12, :task_status)");
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalsDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonPedestalsDat::writeDB(const EcalLogicID* ecid, const MonPedestalsDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonPedestalsDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonPedestalsDat::writeDB:  Bad EcalLogicID")); }
 
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getPedMeanG1());
    m_writeStmt->setFloat(4, item->getPedMeanG6());
    m_writeStmt->setFloat(5, item->getPedMeanG12());
    m_writeStmt->setFloat(6, item->getPedRMSG1());
    m_writeStmt->setFloat(7, item->getPedRMSG6());
    m_writeStmt->setFloat(8, item->getPedRMSG12());
    m_writeStmt->setInt(9, item->getTaskStatus());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalsDat::writeDB():  "+e.getMessage()));
  }
}



void MonPedestalsDat::fetchData(map< EcalLogicID, MonPedestalsDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonPedestalsDat::writeDB:  IOV not in DB")); 
    return;
  }
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.ped_mean_g1, d.ped_mean_g6, d.ped_mean_g12, "
		 "d.ped_rms_g1, d.ped_rms_g6, d.ped_rms_g12, d.task_status "
		 "FROM channelview cv JOIN mon_pedestals_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonPedestalsDat > p;
    MonPedestalsDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setPedMeanG1( rset->getFloat(7) );  
      dat.setPedMeanG6( rset->getFloat(8) );
      dat.setPedMeanG12( rset->getFloat(9) );
      dat.setPedRMSG1( rset->getFloat(10) );
      dat.setPedRMSG6( rset->getFloat(11) );
      dat.setPedRMSG12( rset->getFloat(12) );
      dat.setTaskStatus( rset->getInt(13) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalsDat::fetchData:  "+e.getMessage()));
  }
}
