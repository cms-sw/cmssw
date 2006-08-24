#include <stdexcept>
#include <string>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/MonLaserPulseDat.h"

using namespace std;
using namespace oracle::occi;

MonLaserPulseDat::MonLaserPulseDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_pulseHeightMean = 0;
  m_pulseHeightRMS = 0;
  m_pulseWidthMean = 0;
  m_pulseWidthRMS = 0;
}



MonLaserPulseDat::~MonLaserPulseDat()
{
}



void MonLaserPulseDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_laser_pulse_dat (iov_id, logic_id, "
			"pulse_height_mean, pulse_height_rms, pulse_width_mean, pulse_width_rms) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6)");
  } catch (SQLException &e) {
    throw(runtime_error("MonLaserPulseDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonLaserPulseDat::writeDB(const EcalLogicID* ecid, const MonLaserPulseDat* item, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("MonLaserPulseDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("MonLaserPulseDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getPulseHeightMean() );
    m_writeStmt->setFloat(4, item->getPulseHeightRMS() );
    m_writeStmt->setFloat(5, item->getPulseWidthMean() );
    m_writeStmt->setFloat(6, item->getPulseWidthRMS() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("MonLaserPulseDat::writeDB():  "+e.getMessage()));
  }
}



void MonLaserPulseDat::fetchData(std::map< EcalLogicID, MonLaserPulseDat >* fillMap, MonRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("MonLaserPulseDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.pulse_height_mean, d.pulse_height_rms, d.pulse_width_mean, d.pulse_width_rms "
		 "FROM channelview cv JOIN mon_laser_pulse_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, MonLaserPulseDat > p;
    MonLaserPulseDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setPulseHeightMean( rset->getFloat(7) );
      dat.setPulseHeightRMS( rset->getFloat(8) );
      dat.setPulseWidthMean( rset->getFloat(9) );
      dat.setPulseWidthRMS( rset->getFloat(10) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(runtime_error("MonLaserPulseDat::fetchData():  "+e.getMessage()));
  }
}
