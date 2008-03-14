#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODLaserConfig.h"

using namespace std;
using namespace oracle::occi;

ODLaserConfig::ODLaserConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

   m_ID=0;
   m_wave=0;
   m_power=0;
   m_switch=0;
   m_filter=0;
}



ODLaserConfig::~ODLaserConfig()
{
}



void ODLaserConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_Laser_CONFIGURATION ( "
			"wavelength, power_setting, optical_switch, filter ) "
			"VALUES (  :1, :2, :3, :4 )");
  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODLaserConfig::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getWaveLength());
    m_writeStmt->setInt(2, this->getPower());
    m_writeStmt->setInt(3, this->getOpticalSwitch());
    m_writeStmt->setInt(4, this->getFilter());
    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODLaserConfig::writeDB:  Failed to write"));
  }


}


void ODLaserConfig::clear(){
   m_wave=0;
   m_power=0;
   m_switch=0;
   m_filter=0;
}


void ODLaserConfig::fetchData(ODLaserConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0){
    throw(runtime_error("ODLaserConfig::fetchData(): no Id defined for this ODLaserConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT d.wavelength, d.power_setting, d.optical_switch, d.filter "
		       "FROM ECAL_Laser_CONFIGURATION d "
		       " where laser_configuration_id = :1 " );
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setWaveLength(   rset->getInt(1) );
    result->setPower(        rset->getInt(2) );
    result->setOpticalSwitch(rset->getInt(3) );
    result->setFilter(       rset->getInt(4) );
  
  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::fetchData():  "+e.getMessage()));
  }
}

int ODLaserConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT laser_configuration_id FROM ecal_laser_configuration "
                 "WHERE wavelength=:1 AND power_setting=:2 AND optical_switch=:3 AND filter=:4   " );
    stmt->setInt(1, getWaveLength());
    stmt->setInt(2, getPower());
    stmt->setInt(3, getOpticalSwitch());
    stmt->setInt(4, getFilter());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODLaserConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
