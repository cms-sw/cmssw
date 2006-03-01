#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonPNPedDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

MonPNPedDat::MonPNPedDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_pedMeanG1 = 0;
  m_pedRMSG1 = 0;
  m_pedMeanG16 = 0;
  m_pedRMSG16 = 0;
  m_taskStatus = 0;
}



MonPNPedDat::~MonPNPedDat()
{
}



void MonPNPedDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_pn_ped_dat (iov_id, logic_id, "
			"ped_mean_g1, ped_rms_g1, ped_mean_g16, ped_rms_g16, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6, :7)");
  } catch (SQLException &e) {
    throw(runtime_error("MonPNPedDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonPNPedDat::writeDB(const EcalLogicID* ecid, const MonPNPedDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonPNPedDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonPNPedDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getPedMeanG1() );
    m_writeStmt->setFloat(4, item->getPedRMSG1() );
    m_writeStmt->setFloat(5, item->getPedMeanG16() );
    m_writeStmt->setFloat(6, item->getPedRMSG16() );
    m_writeStmt->setInt(7, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonPNPedDat::writeDB():  "+e.getMessage()));
  }
}



void MonPNPedDat::fetchData(std::map< EcalLogicID, MonPNPedDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonPNPedDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.ped_mean_g1,d.ped_rms_g1, d.ped_mean_g16, d.ped_rms_g16, d.task_status "
		 "FROM channelview cv JOIN mon_pn_ped_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonPNPedDat > p;
    MonPNPedDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setPedMeanG1( rset->getFloat(7) );
      dat.setPedRMSG1( rset->getFloat(8) );
      dat.setPedMeanG16( rset->getFloat(9) );
      dat.setPedRMSG16( rset->getFloat(10) );
      dat.setTaskStatus( rset->getInt(11) );
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonPNPedDat::fetchData():  "+e.getMessage()));
  }
}
