#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunLaserRunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunLaserRunDat::RunLaserRunDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_laserSeqType = "";
  m_laserSeqCond = "";
}



RunLaserRunDat::~RunLaserRunDat()
{
}



void RunLaserRunDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO run_laserrun_config_dat (iov_id, logic_id, "
			"laser_sequence_type, laser_sequence_cond) "
			"VALUES (:1, :2, "
			":3, :4 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("RunLaserRunDat::prepareWrite():  "+e.getMessage()));
  }
}



void RunLaserRunDat::writeDB(const EcalLogicID* ecid, const RunLaserRunDat* item, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("RunLaserRunDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("RunLaserRunDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setString(3, item->getLaserSequenceType());
    m_writeStmt->setString(4, item->getLaserSequenceCond());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("RunLaserRunDat::writeDB():  "+e.getMessage()));
  }
}



void RunLaserRunDat::fetchData(map< EcalLogicID, RunLaserRunDat >* fillMap, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("RunLaserRunDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.laser_sequence_type, d.laser_sequence_cond "
		 "FROM channelview cv JOIN run_laserrun_config_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, RunLaserRunDat > p;
    RunLaserRunDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setLaserSequenceType( rset->getString(7));    // maps_to
      dat.setLaserSequenceCond( rset->getString(8));    // maps_to

      p.second = dat;
      fillMap->insert(p);
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunLaserRunDat::fetchData():  "+e.getMessage()));
  }
}
