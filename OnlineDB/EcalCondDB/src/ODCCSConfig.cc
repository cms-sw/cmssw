#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODCCSConfig.h"

using namespace std;
using namespace oracle::occi;

ODCCSConfig::ODCCSConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

   m_ID=0;
   m_daccal=0;
   m_delay=0;
   m_gain=0;
   m_memgain=0;
   m_offset_high=0;
   m_offset_low=0;
   m_offset_mid=0;
   m_pedestal_offset_release="";
   m_system="";
   m_trg_mode="";
   m_trg_filter="";

}



ODCCSConfig::~ODCCSConfig()
{
}



void ODCCSConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_CCS_CONFIGURATION ( "
			"daccal, delay, gain, memgain, offset_high,offset_low,offset_mid, pedestal_offset_release, system, trg_mode, trg_filter) "
			"VALUES (  "
			":daccal, :delay, :gain, :memgain, :offset_high,:offset_low,"
			" :offset_mid, :pedestal_offset_release, :system, :trg_mode, :trg_filter)");
  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::prepareWrite():  "+e.getMessage()));
  }
}



void ODCCSConfig::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  try {

    m_writeStmt->setInt(1, this->getDaccal());
    m_writeStmt->setInt(2, this->getDelay());
    m_writeStmt->setInt(3, this->getGain());
    m_writeStmt->setInt(4, this->getMemGain());
    m_writeStmt->setInt(5, this->getOffsetHigh());
    m_writeStmt->setInt(6, this->getOffsetLow());
    m_writeStmt->setInt(7, this->getOffsetMid());
    m_writeStmt->setString(8, this->getPedestalOffsetRelease());
    m_writeStmt->setString(9, this->getSystem() );
    m_writeStmt->setString(10, this->getTrgMode() );
    m_writeStmt->setString(11, this->getTrgFilter() );

    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODCCSConfig::writeDB:  Failed to write"));
  }


}


void ODCCSConfig::clear(){
   m_daccal=0;
   m_delay=0;
   m_gain=0;
   m_memgain=0;
   m_offset_high=0;
   m_offset_low=0;
   m_offset_mid=0;
   m_pedestal_offset_release="";
   m_system="";
   m_trg_mode="";
   m_trg_filter="";
}


void ODCCSConfig::fetchData(ODCCSConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0){
    throw(runtime_error("ODCCSConfig::fetchData(): no Id defined for this ODCCSConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT d.daccal, d.delay, d.gain, d.memgain, d.offset_high,d.offset_low,"
		       " d.offset_mid, d.pedestal_offset_release, d.system, d.trg_mode, d.trg_filter   "
		       "FROM ECAL_CCS_CONFIGURATION d "
		       " where ccs_configuration_id = :1 " );
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setDaccal(       rset->getInt(1) );
    result->setDelay(        rset->getInt(2) );
    result->setGain(         rset->getInt(3) );
    result->setMemGain(      rset->getInt(4) );
    result->setOffsetHigh(   rset->getInt(5) );
    result->setOffsetLow(    rset->getInt(6) );
    result->setOffsetMid(    rset->getInt(7) );
    result->setPedestalOffsetRelease( rset->getString(8) );
    result->setSystem(       rset->getString(9) );
    result->setTrgMode(      rset->getString(10) );
    result->setTrgFilter(    rset->getString(11) );


  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::fetchData():  "+e.getMessage()));
  }
}

int ODCCSConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT ccs_configuration_id FROM ecal_ccs_configuration "
                 "WHERE   daccal=:daccal AND delay=:delay AND gain=:gain AND memgain=:memgain AND "
                         " OFFSET_HIGH=:offset_high AND OFFSET_LOW=:offset_low AND"
			" OFFSET_MID=:offset_mid AND pedestal_offset_release=:pedestal_offset_release"
                         " AND system=:system AND TRG_MODE=:trg_mode AND trg_filter=:trg_filter " );

    stmt->setInt(1, getDaccal());
    stmt->setInt(2, getDelay());
    stmt->setInt(3, getGain());
    stmt->setInt(4, getMemGain());
    stmt->setInt(5, getOffsetHigh());
    stmt->setInt(6, getOffsetLow());
    stmt->setInt(7, getOffsetMid());
    stmt->setString(8, getPedestalOffsetRelease());
    stmt->setString(9, getSystem() );
    stmt->setString(10, getTrgMode() );
    stmt->setString(11, getTrgFilter() );

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODCCSConfig::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}
