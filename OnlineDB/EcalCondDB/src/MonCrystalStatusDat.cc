#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonCrystalStatusDat.h"

using namespace std;
using namespace oracle::occi;

MonCrystalStatusDat::MonCrystalStatusDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_statusG1 = MonCrystalStatusDef();
  m_statusG6 = MonCrystalStatusDef();
  m_statusG12 = MonCrystalStatusDef();

}



MonCrystalStatusDat::~MonCrystalStatusDat()
{
}



void MonCrystalStatusDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_crystal_status_dat (iov_id, logic_id, "
			"status_g1, status_g6, status_g12) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5) ");
  } catch (SQLException &e) {
    throw(runtime_error("MonCrystalStatusDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonCrystalStatusDat::writeDB(const EcalLogicID* ecid, const MonCrystalStatusDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonCrystalStatusDat::writeDB:  IOV not in DB")); }

  // XXX this is ugly and inefficient
  MonCrystalStatusDef statusDef;

  statusDef = item->getStatusG1();
  statusDef.setConnection(m_env, m_conn);
  int idG1 = statusDef.fetchID();
 
  statusDef = item->getStatusG6();
  statusDef.setConnection(m_env, m_conn);
  int idG6 = statusDef.fetchID();
  
  statusDef = item->getStatusG12();
  statusDef.setConnection(m_env, m_conn);
  int idG12 = statusDef.fetchID();


  if (!(idG1 && idG6 && idG12)) { throw(runtime_error("MonCrystalStatusDat::writeDB:  Status definition not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonCrystalStatusDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, idG1);
    m_writeStmt->setInt(4, idG6);
    m_writeStmt->setInt(5, idG12);

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonCrystalStatusDat::writeDB():  "+e.getMessage()));
  }
}



void MonCrystalStatusDat::fetchData(map< EcalLogicID, MonCrystalStatusDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonCrystalStatusDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.status_g1, d.status_g6, d.status_g12 "
		 "FROM channelview cv JOIN mon_crystal_status_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonCrystalStatusDat > p;
    MonCrystalStatusDat dat;
    MonCrystalStatusDef statusDef;
    statusDef.setConnection(m_env, m_conn);
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      statusDef.setByID( rset->getInt(7) );
      dat.setStatusG1( statusDef );
      statusDef.setByID( rset->getInt(8) );
      dat.setStatusG6( statusDef );
      statusDef.setByID( rset->getInt(9) );
      dat.setStatusG12( statusDef );


      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonCrystalStatusDat::fetchData():  "+e.getMessage()));
  }
}
