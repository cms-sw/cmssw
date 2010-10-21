#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <boost/lexical_cast.hpp>

#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunCrystalErrorsDat::RunCrystalErrorsDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_errorBits = 0;
}



RunCrystalErrorsDat::~RunCrystalErrorsDat()
{
}



void RunCrystalErrorsDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    /* Using TO_NUMBER because OCCI does not support 64-bit integers well */
    m_writeStmt->setSQL("INSERT INTO run_crystal_errors_dat (iov_id, logic_id, "
			"error_bits) "
			"VALUES (:iov_id, :logic_id, "
			"to_number(:error_bits))");
  } catch (SQLException &e) {
    throw(std::runtime_error("RunCrystalErrorsDat::prepareWrite():  "+e.getMessage()));
  }
}



void RunCrystalErrorsDat::writeDB(const EcalLogicID* ecid, const RunCrystalErrorsDat* item, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("RunCrystalErrorsDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("RunCrystalErrorsDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setString(3, ( boost::lexical_cast<std::string>(item->getErrorBits()) ).c_str());
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("RunCrystalErrorsDat::writeDB():  "+e.getMessage()));
  }
}



void RunCrystalErrorsDat::fetchData(map< EcalLogicID, RunCrystalErrorsDat >* fillMap, RunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("RunCrystalErrorsDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    /* Using TO_CHAR because OCCI does not support 64-bit integers well */
    stmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "to_char(d.error_bits) "
		 "FROM channelview cv JOIN run_crystal_errors_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();
    
    std::pair< EcalLogicID, RunCrystalErrorsDat > p;
    RunCrystalErrorsDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setErrorBits( boost::lexical_cast<uint64_t>(rset->getString(7)) );

      p.second = dat;
      fillMap->insert(p);
    }
    m_conn->terminateStatement(stmt);

  } catch (SQLException &e) {
    throw(std::runtime_error("RunCrystalErrorsDat::fetchData():  "+e.getMessage()));
  }
}
