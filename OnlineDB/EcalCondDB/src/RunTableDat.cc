#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/RunTableDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunTableDat::RunTableDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_table_x = 0;
  m_table_y = 0;
  m_numSpills = 0;
  m_numEvents = 0;
}



RunTableDat::~RunTableDat()
{
}



void RunTableDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO run_h4table_position_dat (iov_id, logic_id, "
			"table_x, table_y, number_of_spills, number_of_events ) "
			"VALUES (:iov_id, :logic_id, "
			":table_x, :table_y, :number_of_spills, :number_of_events)");
  } catch (SQLException &e) {
    throw(runtime_error("RunTableDat::prepareWrite():  "+e.getMessage()));
  }
}



void RunTableDat::writeDB(const EcalLogicID* ecid, const RunTableDat* item, RunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("RunTableDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("RunTableDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getTableX());
    m_writeStmt->setInt(4, item->getTableY());
    m_writeStmt->setInt(5, item->getNumSpills());
    m_writeStmt->setInt(6, item->getNumEvents());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("RunTableDat::writeDB():  "+e.getMessage()));
  }
}



void RunTableDat::fetchData(map< EcalLogicID, RunTableDat >* fillMap, RunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("RunTableDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.table_x, d.table_y, d.number_of_spills, d.number_of_events "
		 "FROM channelview cv JOIN run_h4table_position_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, RunTableDat > p;
    RunTableDat dat;
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
    throw(runtime_error("RunTableDat::fetchData():  "+e.getMessage()));
  }
}
