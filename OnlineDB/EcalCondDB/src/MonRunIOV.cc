#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

MonRunIOV::MonRunIOV()
{
  m_conn = NULL;
  m_ID = 0;
  m_monRunTag = MonRunTag();
  m_runIOV = RunIOV();
  m_subRunNum = 0;
  m_subRunStart = Tm();
  m_subRunEnd = Tm();
}



MonRunIOV::~MonRunIOV()
{
}


void MonRunIOV::setID(int id)
{
    m_ID = id;
}

void MonRunIOV::setMonRunTag(MonRunTag tag)
{
  if (tag != m_monRunTag) {
    m_ID = 0;
    m_monRunTag = tag;
  }
}



MonRunTag MonRunIOV::getMonRunTag() const
{
  return m_monRunTag;
}



void MonRunIOV::setRunIOV(RunIOV iov)
{
  if (iov != m_runIOV) {
    m_ID = 0;
    m_runIOV = iov;
  }
}

RunIOV MonRunIOV::getRunIOV() 
{ 
  return m_runIOV;
}


void MonRunIOV::setSubRunNumber(subrun_t subrun)
{
  if (subrun != m_subRunNum) {
    m_ID = 0;
    m_subRunNum = subrun;
  }
}



run_t MonRunIOV::getSubRunNumber() const
{
  return m_subRunNum;
}



void MonRunIOV::setSubRunStart(Tm start)
{
  if (start != m_subRunStart) {
    m_ID = 0;
    m_subRunStart = start;
  }
}



Tm MonRunIOV::getSubRunStart() const
{
  return m_subRunStart;
}



void MonRunIOV::setSubRunEnd(Tm end)
{
  if (end != m_subRunEnd) {
    m_ID = 0;
    m_subRunEnd = end;
  }
}



Tm MonRunIOV::getSubRunEnd() const
{
  return m_subRunEnd;
}



int MonRunIOV::fetchID()
  throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  // fetch the parent IDs
  int monRunTagID, runIOVID;
  this->fetchParentIDs(&monRunTagID, &runIOVID);

  if (!monRunTagID || !runIOVID) { 
    return 0;
  }

  DateHandler dh(m_env, m_conn);

  if (m_subRunEnd.isNull()) {
    m_subRunEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT iov_id FROM mon_run_iov "
		 "WHERE tag_id = :1 AND "
		 "run_iov_id   = :2 AND "
		 "subrun_num   = :3 AND "
		 "subrun_start = :4 AND "
		 "subrun_end   = :5");
    stmt->setInt(1, monRunTagID);
    stmt->setInt(2, runIOVID);
    stmt->setInt(3, m_subRunNum);
    stmt->setDate(4, dh.tmToDate(m_subRunStart));
    stmt->setDate(5, dh.tmToDate(m_subRunEnd));
  
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("MonRunIOV::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void MonRunIOV::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT tag_id, run_iov_id, subrun_num, subrun_start, subrun_end FROM mon_run_iov WHERE iov_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       int monRunTagID = rset->getInt(1);
       int runIOVID = rset->getInt(2);
       m_subRunNum = rset->getInt(3);
       Date startDate = rset->getDate(4);
       Date endDate = rset->getDate(5);
	 
       m_subRunStart = dh.dateToTm( startDate );
       m_subRunEnd = dh.dateToTm( endDate );

       m_monRunTag.setConnection(m_env, m_conn);
       m_monRunTag.setByID(monRunTagID);

       m_runIOV.setConnection(m_env, m_conn);
       m_runIOV.setByID(runIOVID);

       m_ID = id;
     } else {
       throw(std::runtime_error("MonRunIOV::setByID:  Given tag_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("MonRunIOV::setByID:  "+e.getMessage()));
   }
}



int MonRunIOV::writeDB()
  throw(std::runtime_error)
{
  this->checkConnection();

  // Check if this IOV has already been written
  if (this->fetchID()) {
    return m_ID;
  }

  // fetch Parent IDs
  int monRunTagID, runIOVID;
  this->fetchParentIDs(&monRunTagID, &runIOVID);
		       
  if (!monRunTagID) {
    monRunTagID = m_monRunTag.writeDB();
  }
  
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  if (m_subRunStart.isNull()) {
    throw(std::runtime_error("MonRunIOV::writeDB:  Must setSubRunStart before writing"));
  }
  
  if (m_subRunEnd.isNull()) {
    m_subRunEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    
    stmt->setSQL("INSERT INTO mon_run_iov (iov_id, tag_id, run_iov_id, subrun_num, subrun_start, subrun_end) "
		 "VALUES (mon_run_iov_sq.NextVal, :1, :2, :3, :4, :5)");
    stmt->setInt(1, monRunTagID);
    stmt->setInt(2, runIOVID);
    stmt->setInt(3, m_subRunNum);
    stmt->setDate(4, dh.tmToDate(m_subRunStart));
    stmt->setDate(5, dh.tmToDate(m_subRunEnd));

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("MonRunIOV::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("MonRunIOV::writeDB:  Failed to write"));
  }
  
  return m_ID;
}



void MonRunIOV::fetchParentIDs(int* monRunTagID, int* runIOVID)
  throw(std::runtime_error)
{
  // get the MonRunTag
  m_monRunTag.setConnection(m_env, m_conn);
  *monRunTagID = m_monRunTag.fetchID();

  // get the RunIOV
  m_runIOV.setConnection(m_env, m_conn);
  *runIOVID = m_runIOV.fetchID();

  if (! *runIOVID) { 
    throw(std::runtime_error("MonRunIOV:  Given RunIOV does not exist in DB")); 
  }

}



void MonRunIOV::setByRun(MonRunTag* montag, RunIOV* runiov, subrun_t subrun)
  throw(std::runtime_error)
{
  this->checkConnection();
  
  runiov->setConnection(m_env, m_conn);
  int runIOVID = runiov->fetchID();

  if (!runIOVID) {
    throw(std::runtime_error("MonRunIOV::setByRun:  Given RunIOV does not exist in DB"));
  }

  montag->setConnection(m_env, m_conn);
  int monTagID = montag->fetchID();
  
  if (!monTagID) {
    throw(std::runtime_error("MonRunIOV::setByRun:  Given MonRunTag does not exist in the DB"));
  }

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT iov_id, subrun_start, subrun_end FROM mon_run_iov "
		 "WHERE tag_id = :1 AND run_iov_id = :2 AND subrun_num = :3");
    stmt->setInt(1, monTagID);
    stmt->setInt(2, runIOVID);
    stmt->setInt(3, subrun);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_monRunTag = *montag;
      m_runIOV = *runiov;
      m_subRunNum = subrun;
      
      m_ID = rset->getInt(1);
      Date startDate = rset->getDate(2);
      Date endDate = rset->getDate(3);
	 
      m_subRunStart = dh.dateToTm( startDate );
      m_subRunEnd = dh.dateToTm( endDate );
    } else {
      throw(std::runtime_error("MonRunIOV::setByRun:  Given subrun is not in the database"));
    }
     
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("MonRunIOV::setByRun:  "+e.getMessage()));
  }
  
}
