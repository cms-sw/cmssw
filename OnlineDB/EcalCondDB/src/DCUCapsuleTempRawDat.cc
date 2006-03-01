#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/DCUCapsuleTempRawDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

DCUCapsuleTempRawDat::DCUCapsuleTempRawDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_capsuleTempADC = 0;
  m_capsuleTempRMS = 0;
}



DCUCapsuleTempRawDat::~DCUCapsuleTempRawDat()
{
}



void DCUCapsuleTempRawDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO dcu_capsule_temp_raw_dat (iov_id, logic_id, "
			"capsule_temp_adc, capsule_temp_rms) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4)");
  } catch (SQLException &e) {
    throw(runtime_error("DCUCapsuleTempRawDat::prepareWrite():  "+e.getMessage()));
  }
}



void DCUCapsuleTempRawDat::writeDB(const EcalLogicID* ecid, const DCUCapsuleTempRawDat* item, DCUIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("DCUCapsuleTempRawDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("DCUCapsuleTempRawDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getCapsuleTempADC() );
    m_writeStmt->setFloat(4, item->getCapsuleTempRMS() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("DCUCapsuleTempRawDat::writeDB():  "+e.getMessage()));
  }
}



void DCUCapsuleTempRawDat::fetchData(std::map< EcalLogicID, DCUCapsuleTempRawDat >* fillMap, DCUIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("DCUCapsuleTempRawDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.capsule_temp_adc, d.capsule_temp_rms "
		 "FROM channelview cv JOIN dcu_capsule_temp_raw_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, DCUCapsuleTempRawDat > p;
    DCUCapsuleTempRawDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setCapsuleTempADC( rset->getFloat(7) );
      dat.setCapsuleTempRMS( rset->getFloat(8) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("DCUCapsuleTempRawDat::fetchData():  "+e.getMessage()));
  }
}
