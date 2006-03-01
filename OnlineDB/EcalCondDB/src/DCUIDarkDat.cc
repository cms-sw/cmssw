#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/DCUIDarkDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

DCUIDarkDat::DCUIDarkDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_apdIDark = 0;
}



DCUIDarkDat::~DCUIDarkDat()
{
}



void DCUIDarkDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO dcu_idark_dat (iov_id, logic_id, "
			"apd_idark) "
			"VALUES (:iov_id, :logic_id, "
			":apd_idark)");
  } catch (SQLException &e) {
    throw(runtime_error("DCUIDarkDat::prepareWrite():  "+e.getMessage()));
  }
}



void DCUIDarkDat::writeDB(const EcalLogicID* ecid, const DCUIDarkDat* item, DCUIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("DCUIDarkDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("DCUIDarkDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getAPDIDark() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("DCUIDarkDat::writeDB():  "+e.getMessage()));
  }
}



void DCUIDarkDat::fetchData(std::map< EcalLogicID, DCUIDarkDat >* fillMap, DCUIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("DCUIDarkDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.apd_idark "
		 "FROM channelview cv JOIN dcu_idark_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, DCUIDarkDat > p;
    DCUIDarkDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setAPDIDark( rset->getFloat(7) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("DCUIDarkDat::fetchData():  "+e.getMessage()));
  }
}
