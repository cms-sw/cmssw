#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonPedestalsOnlineDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

using namespace std;
using namespace oracle::occi;

MonPedestalsOnlineDat::MonPedestalsOnlineDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_adcMeanG12 = 0;
  m_adcRMSG12 = 0;
  m_taskStatus = 0;
}



MonPedestalsOnlineDat::~MonPedestalsOnlineDat()
{
}



void MonPedestalsOnlineDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_pedestals_online_dat (iov_id, logic_id, "
			"adc_mean_g12, adc_rms_g12, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":adc_mean_g12, :adc_rms_g12, :task_status)");
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalsOnlineDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonPedestalsOnlineDat::writeDB(const EcalLogicID* ecid, const MonPedestalsOnlineDat* item, MonRunIOV* iov )
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonPedestalsOnlineDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonPedestalsOnlineDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getADCMeanG12() );
    m_writeStmt->setFloat(4, item->getADCRMSG12() );
    m_writeStmt->setInt(5, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalsOnlineDat::writeDB():  "+e.getMessage()));
  }
}



void MonPedestalsOnlineDat::fetchData(std::map< EcalLogicID, MonPedestalsOnlineDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonPedestalsOnlineDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.adc_mean_g12, d.adc_rms_g12, d.task_status "
		 "FROM channelview cv JOIN mon_pedestals_online_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonPedestalsOnlineDat > p;
    MonPedestalsOnlineDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setADCMeanG12( rset->getFloat(7) );
      dat.setADCRMSG12( rset->getFloat(8) );
      dat.setTaskStatus( rset->getInt(9) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonPedestalsOnlineDat::fetchData():  "+e.getMessage()));
  }
}
