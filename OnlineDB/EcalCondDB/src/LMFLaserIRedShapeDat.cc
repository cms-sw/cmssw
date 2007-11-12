#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFLaserIRedShapeDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

LMFLaserIRedShapeDat::LMFLaserIRedShapeDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_alpha = 0;
  m_alphaRMS = 0;
  m_beta = 0;
  m_betaRMS = 0;
}



LMFLaserIRedShapeDat::~LMFLaserIRedShapeDat()
{
}



void LMFLaserIRedShapeDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_laser_ired_shape_dat (iov_id, logic_id, "
			"alpha, alpha_rms, beta, beta_rms) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6)");
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserIRedShapeDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFLaserIRedShapeDat::writeDB(const EcalLogicID* ecid, const LMFLaserIRedShapeDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFLaserIRedShapeDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFLaserIRedShapeDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getAlpha() );
    m_writeStmt->setFloat(4, item->getAlphaRMS() );

    m_writeStmt->setFloat(5, item->getBeta() );
    m_writeStmt->setFloat(6, item->getBetaRMS() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserIRedShapeDat::writeDB():  "+e.getMessage()));
  }
}



void LMFLaserIRedShapeDat::fetchData(std::map< EcalLogicID, LMFLaserIRedShapeDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFLaserIRedShapeDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.alpha, d.alpha_rms, d.beta, d.beta_rms "
		 "FROM channelview cv JOIN lmf_laser_ired_shape_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, LMFLaserIRedShapeDat > p;
    LMFLaserIRedShapeDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setAlpha( rset->getFloat(7) );
      dat.setAlphaRMS( rset->getFloat(8) );
      dat.setBeta( rset->getFloat(9) );
      dat.setBetaRMS( rset->getFloat(10) );

      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserIRedShapeDat::fetchData():  "+e.getMessage()));
  }
}
