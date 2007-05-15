#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFPNTestPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

LMFPNTestPulseDat::LMFPNTestPulseDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_adcMean = 0;
  m_adcRMS = 0;
}



LMFPNTestPulseDat::~LMFPNTestPulseDat()
{
}



void LMFPNTestPulseDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_pn_test_pulse_dat (iov_id, logic_id, "
			"adc_mean, adc_rms) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4)");
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNTestPulseDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFPNTestPulseDat::writeDB(const EcalLogicID* ecid, const LMFPNTestPulseDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFPNTestPulseDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFPNTestPulseDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getADCMean() );
    m_writeStmt->setFloat(4, item->getADCRMS() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNTestPulseDat::writeDB():  "+e.getMessage()));
  }
}



void LMFPNTestPulseDat::fetchData(std::map< EcalLogicID, LMFPNTestPulseDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFPNTestPulseDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.adc_mean, d.adc_rms "
		 "FROM channelview cv JOIN lmf_pn_test_pulse_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, LMFPNTestPulseDat > p;
    LMFPNTestPulseDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setADCMean( rset->getFloat(7) );
      dat.setADCRMS( rset->getFloat(8) );

      p.second = dat;
      fillMap->insert(p);
    }


  } catch (SQLException &e) {
    throw(runtime_error("LMFPNTestPulseDat::fetchData():  "+e.getMessage()));
  }
}

//  LocalWords:  EcalCondDB
