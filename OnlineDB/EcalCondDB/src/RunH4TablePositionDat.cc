#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunH4TablePositionDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunH4TablePositionDat::RunH4TablePositionDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_table_x = 0;
  m_table_y = 0;
  m_numSpills = 0;
  m_numEvents = 0;
}



RunH4TablePositionDat::~RunH4TablePositionDat()
{
}



void RunH4TablePositionDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO run_h4_table_position_dat (iov_id, logic_id, "
			"table_x, table_y, number_of_spills, number_of_events ) "
			"VALUES (:iov_id, :logic_id, "
			":table_x, :table_y, :number_of_spills, :number_of_events)");
  } catch (SQLException &e) {
    throw(std::runtime_error("RunH4TablePositionDat::prepareWrite():  "+e.getMessage()));
  }
}



void RunH4TablePositionDat::writeDB(const EcalLogicID* ecid, const RunH4TablePositionDat* item, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("RunH4TablePositionDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("RunH4TablePositionDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getTableX());
    m_writeStmt->setInt(4, item->getTableY());
    m_writeStmt->setInt(5, item->getNumSpills());
    m_writeStmt->setInt(6, item->getNumEvents());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("RunH4TablePositionDat::writeDB():  "+e.getMessage()));
  }
}



void RunH4TablePositionDat::fetchData(map< EcalLogicID, RunH4TablePositionDat >* fillMap, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("RunH4TablePositionDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.table_x, d.table_y, d.number_of_spills, d.number_of_events "
		 "FROM channelview cv JOIN run_h4_table_position_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, RunH4TablePositionDat > p;
    RunH4TablePositionDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setTableX( rset->getInt(7) );
      dat.setTableY( rset->getInt(8) );
      dat.setNumSpills( rset->getInt(9) );
      dat.setNumEvents( rset->getInt(10) );

      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(std::runtime_error("RunH4TablePositionDat::fetchData():  "+e.getMessage()));
  }
}
