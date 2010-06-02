#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFRunDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"

using namespace std;
using namespace oracle::occi;

LMFRunDat::LMFRunDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  //
  m_numEvents = 0;
  m_status = 0;
}



LMFRunDat::~LMFRunDat()
{
}



void LMFRunDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_run_dat (lmf_iov_id, logic_id, "
			"nevents, quality_flag ) "
			"VALUES (:lmf_iov_id, :logic_id, "
			":nevents, :quality_flag ) ");
  } catch (SQLException &e) {
    throw(runtime_error("LMFRunDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFRunDat::writeDB(const EcalLogicID* ecid, const LMFRunDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFRunDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFRunDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getNumEvents());
    m_writeStmt->setInt(4, item->getQualityFlag());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFRunDat::writeDB():  "+e.getMessage()));
  }
}



void LMFRunDat::fetchData(map< EcalLogicID, LMFRunDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFRunDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.nevents, d.quality_flag "
		 "FROM channelview cv JOIN lmf_run_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.lmf_iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, LMFRunDat > p;
    LMFRunDat dat;

    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setNumEvents( rset->getInt(7) );
      dat.setQualityFlag(  rset->getInt(8) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("LMFRunDat::fetchData():  "+e.getMessage()));
  }
}
