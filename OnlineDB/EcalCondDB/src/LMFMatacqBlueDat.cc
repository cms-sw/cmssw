#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFMatacqBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

LMFMatacqBlueDat::LMFMatacqBlueDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_amplitude = 0;
  m_timeoffset = 0;
  m_width = 0;
}



LMFMatacqBlueDat::~LMFMatacqBlueDat()
{
}



void LMFMatacqBlueDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_matacq_blue_dat (iov_id, logic_id, "
			"amplitude, width, timeoffset) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5)");
  } catch (SQLException &e) {
    throw(runtime_error("LMFMatacqBlueDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFMatacqBlueDat::writeDB(const EcalLogicID* ecid, const LMFMatacqBlueDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFMatacqBlueDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFMatacqBlueDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getAmplitude() );
    m_writeStmt->setFloat(4, item->getWidth() );
    m_writeStmt->setFloat(5, item->getTimeOffset() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFMatacqBlueDat::writeDB():  "+e.getMessage()));
  }
}



void LMFMatacqBlueDat::fetchData(std::map< EcalLogicID, LMFMatacqBlueDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFMatacqBlueDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.amplitude, d.width, d.timeoffset "
		 "FROM channelview cv JOIN lmf_matacq_blue_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, LMFMatacqBlueDat > p;
    LMFMatacqBlueDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setAmplitude( rset->getFloat(7) );
      dat.setWidth( rset->getFloat(8) );
      dat.setTimeOffset( rset->getFloat(9) );

      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFMatacqBlueDat::fetchData():  "+e.getMessage()));
  }
}
