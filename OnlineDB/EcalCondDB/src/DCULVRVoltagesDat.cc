#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/DCULVRVoltagesDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

DCULVRVoltagesDat::DCULVRVoltagesDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_vfe1_A = 0;
  m_vfe2_A = 0;
  m_vfe3_A = 0;
  m_vfe4_A = 0;
  m_vfe5_A = 0;
  m_VCC = 0;
  m_vfe4_5_D = 0;
  m_vfe1_2_3_D = 0;
  m_buffer = 0;
  m_fenix = 0;
  m_V43_A = 0;
  m_OCM = 0;
  m_GOH = 0;
  m_INH = 0;
  m_V43_D = 0;
}



DCULVRVoltagesDat::~DCULVRVoltagesDat()
{
}



void DCULVRVoltagesDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO dcu_lvr_voltages_dat (iov_id, logic_id, "
			"vfe1_A, vfe2_A, vfe3_A, vfe4_A, vfe5_A, VCC, vfe4_5_D, vfe1_2_3_D, buffer, fenix, V43_A, OCM, GOH, INH, V43_D) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17)");
  } catch (SQLException &e) {
    throw(runtime_error("DCULVRVoltagesDat::prepareWrite():  "+e.getMessage()));
  }
}



void DCULVRVoltagesDat::writeDB(const EcalLogicID* ecid, const DCULVRVoltagesDat* item, DCUIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("DCULVRVoltagesDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("DCULVRVoltagesDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getVFE1_A() );
    m_writeStmt->setFloat(4, item->getVFE2_A() );
    m_writeStmt->setFloat(5, item->getVFE3_A() );
    m_writeStmt->setFloat(6, item->getVFE4_A() );
    m_writeStmt->setFloat(7, item->getVFE5_A() );
    m_writeStmt->setFloat(8, item->getVCC() );
    m_writeStmt->setFloat(9, item->getVFE4_5_D() );
    m_writeStmt->setFloat(10, item->getVFE1_2_3_D() );
    m_writeStmt->setFloat(11, item->getBuffer() );
    m_writeStmt->setFloat(12, item->getFenix() );
    m_writeStmt->setFloat(13, item->getV43_A() );
    m_writeStmt->setFloat(14, item->getOCM() );
    m_writeStmt->setFloat(15, item->getGOH() );
    m_writeStmt->setFloat(16, item->getINH() );
    m_writeStmt->setFloat(17, item->getV43_D() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("DCULVRVoltagesDat::writeDB():  "+e.getMessage()));
  }
}



void DCULVRVoltagesDat::fetchData(std::map< EcalLogicID, DCULVRVoltagesDat >* fillMap, DCUIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("DCULVRVoltagesDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.vfe1_A, d.vfe2_A, d.vfe3_A, d.vfe4_A, d.vfe5_A, d.VCC, d.vfe4_5_D, d.vfe1_2_3_D, d.buffer, d.fenix, d.V43_A, d.OCM, d.GOH, d.INH, d.V43_D  "
		 "FROM channelview cv JOIN dcu_lvr_voltages_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, DCULVRVoltagesDat > p;
    DCULVRVoltagesDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setVFE1_A( rset->getFloat(7) );
      dat.setVFE2_A( rset->getFloat(8) );
      dat.setVFE3_A( rset->getFloat(9) );
      dat.setVFE4_A( rset->getFloat(10) );
      dat.setVFE5_A( rset->getFloat(11) );
      dat.setVCC( rset->getFloat(12) );
      dat.setVFE4_5_D( rset->getFloat(13) );
      dat.setVFE1_2_3_D( rset->getFloat(14) );
      dat.setBuffer( rset->getFloat(15) );
      dat.setFenix( rset->getFloat(16) );
      dat.setV43_A( rset->getFloat(17) );
      dat.setOCM( rset->getFloat(18) );
      dat.setGOH( rset->getFloat(19) );
      dat.setINH( rset->getFloat(20) );
      dat.setV43_D( rset->getFloat(21) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("DCULVRVoltagesDat::fetchData():  "+e.getMessage()));
  }
}
