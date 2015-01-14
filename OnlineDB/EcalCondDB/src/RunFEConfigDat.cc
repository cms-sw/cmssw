#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunFEConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunFEConfigDat::RunFEConfigDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_config = 0;

}



RunFEConfigDat::~RunFEConfigDat()
{
}



void RunFEConfigDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO run_FEConfig_dat (iov_id, logic_id, "
			"Config_id ) "
			"VALUES (:iov_id, :logic_id, "
			":Config_id ) ");
  } catch (SQLException &e) {
    throw(std::runtime_error("RunFEConfigDat::prepareWrite():  "+e.getMessage()));
  }
}



void RunFEConfigDat::writeDB(const EcalLogicID* ecid, const RunFEConfigDat* item, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("RunFEConfigDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("RunFEConfigDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getConfigId());


    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("RunFEConfigDat::writeDB():  "+e.getMessage()));
  }
}



void RunFEConfigDat::fetchData(map< EcalLogicID, RunFEConfigDat >* fillMap, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("RunFEConfigDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    //    createReadStatement();
    //    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.Config_id "
		 "FROM channelview cv JOIN run_FEConfig_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    //    m_readStmt->setInt(1, iovID);
    //    ResultSet* rset = m_readStmt->executeQuery();
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, RunFEConfigDat > p;
    RunFEConfigDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setConfigId( rset->getInt(7) );
 
      p.second = dat;
      fillMap->insert(p);
    }
    //    terminateReadStatement();
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunFEConfigDat::fetchData():  "+e.getMessage()));
  }
}

