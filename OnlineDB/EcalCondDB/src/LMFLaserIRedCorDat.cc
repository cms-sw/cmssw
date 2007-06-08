#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFLaserIRedCorDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

LMFLaserIRedCorDat::LMFLaserIRedCorDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_apdOverPNAMean = 0;
  m_apdOverPNARMS = 0;
  m_apdOverPNBMean = 0;
  m_apdOverPNMean = 0;
  m_apdOverPNRMS = 0;
}



LMFLaserIRedCorDat::~LMFLaserIRedCorDat()
{
}



void LMFLaserIRedCorDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_laser_ired_cor_dat (iov_id, logic_id, "
			"apd_over_pnA_mean, apd_over_pnA_rms, apd_over_pnB_mean, apd_over_pnB_rms, apd_over_pn_mean, apd_over_pn_rms) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6, :7, :8)");
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserIRedCorDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFLaserIRedCorDat::writeDB(const EcalLogicID* ecid, const LMFLaserIRedCorDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFLaserIRedCorDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFLaserIRedCorDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getAPDOverPNAMean() );
    m_writeStmt->setFloat(4, item->getAPDOverPNARMS() );
    m_writeStmt->setFloat(5, item->getAPDOverPNBMean() );
    m_writeStmt->setFloat(6, item->getAPDOverPNBRMS() );
    m_writeStmt->setFloat(7, item->getAPDOverPNMean() );
    m_writeStmt->setFloat(8, item->getAPDOverPNRMS() );


    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserIRedCorDat::writeDB():  "+e.getMessage()));
  }
}



void LMFLaserIRedCorDat::fetchData(std::map< EcalLogicID, LMFLaserIRedCorDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFLaserIRedCorDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.apd_over_pnA_mean, d.apd_over_pnA_rms, d.apd_over_pnB_mean, d.apd_over_pnB_rms, d.apd_over_pn_mean, d.apd_over_pn_rms "
		 "FROM channelview cv JOIN lmf_laser_ired_cor_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, LMFLaserIRedCorDat > p;
    LMFLaserIRedCorDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setAPDOverPNAMean( rset->getFloat(7) );
      dat.setAPDOverPNARMS( rset->getFloat(8) );
      dat.setAPDOverPNBMean( rset->getFloat(9) );
      dat.setAPDOverPNBRMS( rset->getFloat(10) );
      dat.setAPDOverPNMean( rset->getFloat(11) );
      dat.setAPDOverPNRMS( rset->getFloat(12) );


      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserIRedCorDat::fetchData():  "+e.getMessage()));
  }
}
