#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/LMFPNBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

LMFPNBlueDat::LMFPNBlueDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_pnPeak = 0;
  m_pnErr = 0;
}



LMFPNBlueDat::~LMFPNBlueDat()
{
}



void LMFPNBlueDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_pn_blue_dat (iov_id, logic_id, "
			"pn_peak, pn_err) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4)");
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNBlueDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFPNBlueDat::writeDB(const EcalLogicID* ecid, const LMFPNBlueDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFPNBlueDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFPNBlueDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getPNPeak() );
    m_writeStmt->setFloat(4, item->getPNErr() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNBlueDat::writeDB():  "+e.getMessage()));
  }
}



void LMFPNBlueDat::fetchData(std::map< EcalLogicID, LMFPNBlueDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFPNBlueDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.pn_peak, d.pn_err "
		 "FROM channelview cv JOIN lmf_pn_blue_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, LMFPNBlueDat > p;
    LMFPNBlueDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setPNPeak( rset->getFloat(7) );
      dat.setPNErr( rset->getFloat(8) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNBlueDat::fetchData():  "+e.getMessage()));
  }
}
