#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/CaliHVScanRatioDat.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"

using namespace std;
using namespace oracle::occi;

CaliHVScanRatioDat::CaliHVScanRatioDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_hvratio = 0;
  m_hvratioRMS = 0;
  m_taskStatus = false;
}



CaliHVScanRatioDat::~CaliHVScanRatioDat()
{
}



void CaliHVScanRatioDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();
  
  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO cali_hv_scan_ratio_dat (iov_id, logic_id, "
			"hvratio, hvratio_rms, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5)");
  } catch (SQLException &e) {
    throw(std::runtime_error("CaliHVScanRatioDat::prepareWrite():  "+e.getMessage()));
  }
}



void CaliHVScanRatioDat::writeDB(const EcalLogicID* ecid, const CaliHVScanRatioDat* item, CaliIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();
  
  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("CaliHVScanRatioDat::writeDB:  IOV not in DB")); }
  
  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("CaliHVScanRatioDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    
    m_writeStmt->setFloat(3, item->getHVRatio() );
    m_writeStmt->setFloat(4, item->getHVRatioRMS() );
    m_writeStmt->setInt(5, item->getTaskStatus() );
    
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("CaliHVScanRatioDat::writeDB():  "+e.getMessage()));
  }
}



void CaliHVScanRatioDat::fetchData(std::map< EcalLogicID, CaliHVScanRatioDat >* fillMap, CaliIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();
  
  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("CaliHVScanRatioDat::writeDB:  IOV not in DB")); 
    return;
  }
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.hvratio, d.hvratio_rms, d.task_status "
		 "FROM channelview cv JOIN cali_hv_scan_ratio_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, CaliHVScanRatioDat > p;
    CaliHVScanRatioDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to
      
      dat.setHVRatio( rset->getFloat(7) );
      dat.setHVRatioRMS( rset->getFloat(8) );
      dat.setTaskStatus( rset->getInt(9) );
      
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("CaliHVScanRatioDat::fetchData():  "+e.getMessage()));
  }
}
