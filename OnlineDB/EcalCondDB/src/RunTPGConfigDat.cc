#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunTPGConfigDat::RunTPGConfigDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_config = "";
  m_version=0;
}



RunTPGConfigDat::~RunTPGConfigDat()
{
}



void RunTPGConfigDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO run_TPGConfig_dat (iov_id, logic_id, "
			"Config_tag , version ) "
			"VALUES (:iov_id, :logic_id, "
			":Config_tag , :version ) ");
  } catch (SQLException &e) {
    throw(runtime_error("RunTPGConfigDat::prepareWrite():  "+e.getMessage()));
  }
}



void RunTPGConfigDat::writeDB(const EcalLogicID* ecid, const RunTPGConfigDat* item, RunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("RunTPGConfigDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("RunTPGConfigDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setString(3, item->getConfigTag());
    m_writeStmt->setInt(4, item->getVersion());


    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("RunTPGConfigDat::writeDB():  "+e.getMessage()));
  }
}



void RunTPGConfigDat::fetchData(map< EcalLogicID, RunTPGConfigDat >* fillMap, RunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("RunTPGConfigDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.Config_tag, d.version "
		 "FROM channelview cv JOIN run_TPGConfig_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, RunTPGConfigDat > p;
    RunTPGConfigDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setConfigTag( rset->getString(7) );
      dat.setVersion( rset->getInt(8) );
 

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("RunTPGConfigDat::fetchData():  "+e.getMessage()));
  }
}
