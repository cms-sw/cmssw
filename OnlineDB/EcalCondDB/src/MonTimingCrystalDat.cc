#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonTimingCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

MonTimingCrystalDat::MonTimingCrystalDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_timingMean = 0;
  m_timingRMS = 0;
 }



MonTimingCrystalDat::~MonTimingCrystalDat()
{
}



void MonTimingCrystalDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_timing_crystal_dat (iov_id, logic_id, "
			"timing_mean, timing_rms ) "
			"VALUES (:iov_id, :logic_id, "
			":timing_mean, :timing_rms )");
  } catch (SQLException &e) {
    throw(runtime_error("MonTimingCrystalDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonTimingCrystalDat::writeDB(const EcalLogicID* ecid, const MonTimingCrystalDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonTimingCrystalDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonTimingCrystalDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getTimingMean() );
    m_writeStmt->setFloat(4, item->getTimingRMS() );


    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonTimingCrystalDat::writeDB():  "+e.getMessage()));
  }
}



void MonTimingCrystalDat::fetchData(std::map< EcalLogicID, MonTimingCrystalDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonTimingCrystalDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.timing_mean, d.timing_rms "
		 "FROM channelview cv JOIN mon_timing_crystal_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonTimingCrystalDat > p;
    MonTimingCrystalDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setTimingMean( rset->getFloat(7) );
      dat.setTimingRMS( rset->getFloat(8) );

      p.second = dat;
      fillMap->insert(p);
    }

    m_conn->terminateStatement(stmt);

  } catch (SQLException &e) {
    throw(runtime_error("MonTimingCrystalDat::fetchData():  "+e.getMessage()));
  }
}
