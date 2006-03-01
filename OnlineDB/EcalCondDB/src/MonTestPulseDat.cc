#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonTestPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

using namespace std;
using namespace oracle::occi;

MonTestPulseDat::MonTestPulseDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_adcMeanG1 = 0;
  m_adcRMSG1 = 0;
  m_adcMeanG6 = 0;
  m_adcRMSG6 = 0;
  m_adcMeanG12 = 0;
  m_adcRMSG12 = 0;
  m_taskStatus = 0;
}



MonTestPulseDat::~MonTestPulseDat()
{
}



void MonTestPulseDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_test_pulse_dat (iov_id, logic_id, "
			"adc_mean_g1, adc_rms_g1, adc_mean_g6, adc_rms_g6, adc_mean_g12, adc_rms_g12, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":adc_mean_g1, :adc_rms_g1, :adc_rms_g6, :adc_rms_g6, :adc_mean_g12, :adc_rms_g12, :task_status)");
  } catch (SQLException &e) {
    throw(runtime_error("MonTestPulseDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonTestPulseDat::writeDB(const EcalLogicID* ecid, const MonTestPulseDat* item, MonRunIOV* iov )
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonTestPulseDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonTestPulseDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getADCMeanG1() );
    m_writeStmt->setFloat(4, item->getADCRMSG1() );
    m_writeStmt->setFloat(5, item->getADCMeanG6() );
    m_writeStmt->setFloat(6, item->getADCRMSG6() );
    m_writeStmt->setFloat(7, item->getADCMeanG12() );
    m_writeStmt->setFloat(8, item->getADCRMSG12() );
    m_writeStmt->setInt(9, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonTestPulseDat::writeDB():  "+e.getMessage()));
  }
}



void MonTestPulseDat::fetchData(std::map< EcalLogicID, MonTestPulseDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonTestPulseDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.adc_mean_g1, d.adc_rms_g1, d.adc_mean_g6, d.adc_rms_g6, d.adc_mean_g12, d.adc_rms_g12, d.task_status "
		 "FROM channelview cv JOIN mon_test_pulse_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonTestPulseDat > p;
    MonTestPulseDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setADCMeanG1( rset->getFloat(7) );
      dat.setADCRMSG1( rset->getFloat(8) );
      dat.setADCMeanG6( rset->getFloat(9) );
      dat.setADCRMSG6( rset->getFloat(10) );
      dat.setADCMeanG12( rset->getFloat(11) );
      dat.setADCRMSG12( rset->getFloat(12) );
      dat.setTaskStatus( rset->getInt(13) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonTestPulseDat::fetchData():  "+e.getMessage()));
  }
}
