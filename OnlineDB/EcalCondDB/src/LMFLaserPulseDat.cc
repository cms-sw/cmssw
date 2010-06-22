#include <stdexcept>
#include <cassert>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFLaserPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

int LMFLaserPulseDat::_color = LMFLaserPulseDat::iBlue; //GHM

LMFLaserPulseDat::LMFLaserPulseDat()
{
  assert( _color>=iBlue && _color<=iIRed ); // GHM

  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  //  m_fit_method="";
  m_fit_method = 0;
  m_ampl = 0;
  m_time = 0;
  m_rise = 0;
  m_fwhm = 0;
  m_fw20 = 0;
  m_fw80 = 0;
  m_sliding = 0;

}



LMFLaserPulseDat::~LMFLaserPulseDat()
{
}



void LMFLaserPulseDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  // GHM
  std::string command_ = "INSERT INTO XXXXXX (lmf_iov_id, logic_id, fit_method, mtq_ampl, mtq_time, mtq_rise, mtq_fwhm, mtq_fw20, mtq_fw80, mtq_sliding ) VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10 )";
  command_.replace( command_.find("XXXXXX",0), 6, getTable() );

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL( command_.c_str() );  // GHM
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserPulseDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFLaserPulseDat::writeDB(const EcalLogicID* ecid, const LMFLaserPulseDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFLaserPulseDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFLaserPulseDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    //    m_writeStmt->setString(3, item->getFitMethod() );
    m_writeStmt->setInt(3, item->getFitMethod() );
    m_writeStmt->setFloat(4, item->getAmplitude() );
    m_writeStmt->setFloat(5, item->getTime() );
    m_writeStmt->setFloat(6, item->getRise() );
    m_writeStmt->setFloat(7, item->getFWHM() );
    m_writeStmt->setFloat(8, item->getFW20() );
    m_writeStmt->setFloat(9, item->getFW80() );
    m_writeStmt->setFloat(10, item->getSliding() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserPulseDat::writeDB():  "+e.getMessage()));
  }
}



void LMFLaserPulseDat::fetchData(std::map< EcalLogicID, LMFLaserPulseDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFLaserPulseDat::writeDB:  IOV not in DB")); 
    return;
  }

  // GHM
  std::string command_ = "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, d.fit_method, d.mtq_ampl, d.mtq_time, d.mtq_rise, d.mtq_fwhm, d.mtq_fw20, d.mtq_fw80, d.mtq_sliding FROM channelview cv JOIN XXXXXX d ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to WHERE d.lmf_iov_id = :lmf_iov_id";
  command_.replace( command_.find("XXXXXX",0), 6, getTable() );

  try {

    m_readStmt->setSQL( command_.c_str() );  // GHM
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, LMFLaserPulseDat > p;
    LMFLaserPulseDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      //      dat.setFitMethod( rset->getString(7) );
      dat.setFitMethod( rset->getInt(7) );
      dat.setAmplitude( rset->getFloat(8) );
      dat.setTime( rset->getFloat(9) );
      dat.setRise( rset->getFloat(10) );
      dat.setFWHM( rset->getFloat(11) );
      dat.setFW20( rset->getFloat(12) );
      dat.setFW80( rset->getFloat(13) );
      dat.setSliding( rset->getFloat(14) );
    
      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserPulseDat::fetchData():  "+e.getMessage()));
  }
}

void
LMFLaserPulseDat::setColor( int color ) { _color = color; }
