#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunPTMTempDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunPTMTempDat::RunPTMTempDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_temperature = 0;
}



RunPTMTempDat::~RunPTMTempDat()
{
}



void RunPTMTempDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO run_temperature_ptm_dat (iov_id, logic_id, "
			"temperature) "
			"VALUES (:iov_id, :logic_id, "
			":temperature)");
  } catch (SQLException &e) {
    throw(std::runtime_error("RunPTMTempDat::prepareWrite():  "+e.getMessage()));
  }
}



void RunPTMTempDat::writeDB(const EcalLogicID* ecid, const RunPTMTempDat* item, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("RunPTMTempDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("RunPTMTempDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getTemperature());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("RunPTMTempDat::writeDB():  "+e.getMessage()));
  }
}



void RunPTMTempDat::fetchData(map< EcalLogicID, RunPTMTempDat >* fillMap, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("RunPTMTempDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.temperature "
		 "FROM channelview cv JOIN RUN_TEMPERATURE_PTM_DAT d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, RunPTMTempDat > p;
    RunPTMTempDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setTemperature( rset->getFloat(7) );

      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(std::runtime_error("RunPTMTempDat::fetchData():  "+e.getMessage()));
  }
}
