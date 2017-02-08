#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonLaserStatusDat.h"

using namespace std;
using namespace oracle::occi;

MonLaserStatusDat::MonLaserStatusDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_laserPower = 0;
  m_laserFilter = 0;
  m_laserWavelength = 0;
  m_laserFanout = 0;
}



MonLaserStatusDat::~MonLaserStatusDat()
{
}



void MonLaserStatusDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_laser_status_dat (iov_id, logic_id, "
			"laser_power, laser_filter, laser_wavelength, laser_fanout) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6)");
  } catch (SQLException &e) {
    throw(std::runtime_error("MonLaserStatusDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonLaserStatusDat::writeDB(const EcalLogicID* ecid, const MonLaserStatusDat* item, MonRunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonLaserStatusDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("MonLaserStatusDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getLaserPower() );
    m_writeStmt->setFloat(4, item->getLaserFilter() );
    m_writeStmt->setFloat(5, item->getLaserWavelength() );
    m_writeStmt->setFloat(6, item->getLaserFanout() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("MonLaserStatusDat::writeDB():  "+e.getMessage()));
  }
}



void MonLaserStatusDat::fetchData(std::map< EcalLogicID, MonLaserStatusDat >* fillMap, MonRunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("MonLaserStatusDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.laser_power, d.laser_filter, d.laser_wavelength, d.laser_fanout "
		 "FROM channelview cv JOIN mon_laser_status_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, MonLaserStatusDat > p;
    MonLaserStatusDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setLaserPower( rset->getFloat(7) );
      dat.setLaserFilter( rset->getFloat(8) );
      dat.setLaserWavelength( rset->getFloat(9) );
      dat.setLaserFanout( rset->getFloat(10) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("MonLaserStatusDat::fetchData():  "+e.getMessage()));
  }
}
