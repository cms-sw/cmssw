#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"

using namespace std;
using namespace oracle::occi;

MonOccupancyDat::MonOccupancyDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_eventsOverLowThreshold = 0;
  m_eventsOverHighThreshold = 0;
  m_avgEnergy = 0;
}



MonOccupancyDat::~MonOccupancyDat()
{
}



void MonOccupancyDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_occupancy_dat (iov_id, logic_id, "
			"events_over_low_threshold, events_over_high_threshold, avg_energy) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5)");
  } catch (SQLException &e) {
    throw(runtime_error("MonOccupancyDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonOccupancyDat::writeDB(const EcalLogicID* ecid, const MonOccupancyDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonOccupancyDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonOccupancyDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getEventsOverLowThreshold() );
    m_writeStmt->setInt(4, item->getEventsOverHighThreshold() );
    m_writeStmt->setFloat(5, item->getAvgEnergy() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonOccupancyDat::writeDB():  "+e.getMessage()));
  }
}



void MonOccupancyDat::fetchData(std::map< EcalLogicID, MonOccupancyDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonOccupancyDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.events_over_low_threshold, d.events_over_high_threshold, d.avg_energy "
		 "FROM channelview cv JOIN mon_occupancy_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonOccupancyDat > p;
    MonOccupancyDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setEventsOverLowThreshold( rset->getInt(7) );
      dat.setEventsOverHighThreshold( rset->getInt(8) );
      dat.setAvgEnergy( rset->getFloat(9) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonOccupancyDat::fetchData():  "+e.getMessage()));
  }
}
