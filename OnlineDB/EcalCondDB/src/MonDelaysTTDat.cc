#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonDelaysTTDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

MonDelaysTTDat::MonDelaysTTDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_delayMean = 0;
  m_delayRMS = 0;
  m_taskStatus = 0;
}



MonDelaysTTDat::~MonDelaysTTDat()
{
}



void MonDelaysTTDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_delays_tt_dat (iov_id, logic_id, "
			"delay_mean, delay_rms, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":delay_mean, :delay_rms, :task_status)");
  } catch (SQLException &e) {
    throw(runtime_error("MonDelaysTTDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonDelaysTTDat::writeDB(const EcalLogicID* ecid, const MonDelaysTTDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonDelaysTTDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonDelaysTTDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getDelayMean() );
    m_writeStmt->setFloat(4, item->getDelayRMS() );
    m_writeStmt->setInt(5, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonDelaysTTDat::writeDB():  "+e.getMessage()));
  }
}



void MonDelaysTTDat::fetchData(std::map< EcalLogicID, MonDelaysTTDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonDelaysTTDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.delay_mean, d.delay_rms, d.task_status "
		 "FROM channelview cv JOIN mon_delays_tt_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonDelaysTTDat > p;
    MonDelaysTTDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setDelayMean( rset->getFloat(7) );
      dat.setDelayRMS( rset->getFloat(8) );
      dat.setTaskStatus( rset->getInt(9) );
			

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonDelaysTTDat::fetchData():  "+e.getMessage()));
  }
}
