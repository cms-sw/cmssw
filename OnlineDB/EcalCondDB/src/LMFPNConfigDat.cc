#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFPNConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

LMFPNConfigDat::LMFPNConfigDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_pnAID = 0;
  m_pnBID = 0;
  m_pnAValidity = 0;
  m_pnBValidity = 0;
  m_pnMeanValidity = 0;
}



LMFPNConfigDat::~LMFPNConfigDat()
{
}



void LMFPNConfigDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_pn_config_dat (iov_id, logic_id, "
			"pna_id, pnb_id, pna_validity, pnb_validity, pnmean_validity) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6, :7)");
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNConfigDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFPNConfigDat::writeDB(const EcalLogicID* ecid, const LMFPNConfigDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFPNConfigDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFPNConfigDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getPNAID() );
    m_writeStmt->setInt(4, item->getPNBID() );
    m_writeStmt->setInt(5, item->getPNAValidity() );
    m_writeStmt->setInt(6, item->getPNBValidity() );
    m_writeStmt->setInt(7, item->getPNMeanValidity() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNConfigDat::writeDB():  "+e.getMessage()));
  }
}



void LMFPNConfigDat::fetchData(std::map< EcalLogicID, LMFPNConfigDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFPNConfigDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.pna_id, d.pnb_id, d.pna_validity, d.pnb_validity, d.pnmean_validity "
		 "FROM channelview cv JOIN lmf_pn_config_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, LMFPNConfigDat > p;
    LMFPNConfigDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setPNAID( rset->getInt(7) );
      dat.setPNBID( rset->getInt(8) );
      dat.setPNAValidity( rset->getInt(9) );
      dat.setPNBValidity( rset->getInt(10) );
      dat.setPNMeanValidity( rset->getInt(11) );

      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFPNConfigDat::fetchData():  "+e.getMessage()));
  }
}

//  LocalWords:  EcalCondDB
