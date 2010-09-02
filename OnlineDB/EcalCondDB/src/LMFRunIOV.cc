#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

LMFRunIOV::LMFRunIOV()
{
  m_conn = NULL;
  m_ID = 0;
  m_lmfRunTag = LMFRunTag();
  m_runIOV = RunIOV();
  m_subRunNum = 0;
  m_subRunStart = Tm();
  m_subRunEnd = Tm();
  m_dbtime = Tm();
  m_subrun_type="";
}



LMFRunIOV::~LMFRunIOV()
{
}

void LMFRunIOV::setID(int id)
{
    m_ID = id;
}


void LMFRunIOV::setLMFRunTag(LMFRunTag tag)
{
  if (tag != m_lmfRunTag) {
    m_ID = 0;
    m_lmfRunTag = tag;
  }
}



LMFRunTag LMFRunIOV::getLMFRunTag() const
{
  return m_lmfRunTag;
}

RunIOV LMFRunIOV::getRunIOV() 
{ 
  return m_runIOV;
}


void LMFRunIOV::setRunIOV(RunIOV iov)
{
  if (iov != m_runIOV) {
    m_ID = 0;
    m_runIOV = iov;
  }
}




int LMFRunIOV::fetchID()
  throw(runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  // fetch the parent IDs
  int lmfRunTagID, runIOVID;
  this->fetchParentIDs(&lmfRunTagID, &runIOVID);

  if (!lmfRunTagID || !runIOVID) { 
    return 0;
  }

  DateHandler dh(m_env, m_conn);

  if (m_subRunEnd.isNull()) {
    m_subRunEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT lmf_iov_id FROM lmf_run_iov "
		 "WHERE tag_id = :1 AND "
		 "run_iov_id   = :2 AND "
		 "subrun_num   = :3 AND "
		 "subrun_start = :4 AND "
		 "subrun_end   = :5");
    stmt->setInt(1, lmfRunTagID);
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
    throw(runtime_error("LMFRunIOV::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void LMFRunIOV::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT tag_id, run_iov_id, subrun_num, subrun_start, subrun_end, subrun_type, db_timestamp FROM lmf_run_iov WHERE lmf_iov_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       int lmfRunTagID = rset->getInt(1);
       int runIOVID = rset->getInt(2);
       m_subRunNum = rset->getInt(3);
       Date startDate = rset->getDate(4);
       Date endDate = rset->getDate(5);
       m_subrun_type = rset->getString(6);
       Date dbtime = rset->getDate(7);
	 
       m_subRunStart = dh.dateToTm( startDate );
       m_subRunEnd = dh.dateToTm( endDate );
       m_dbtime = dh.dateToTm(dbtime  );

       m_lmfRunTag.setConnection(m_env, m_conn);
       m_lmfRunTag.setByID(lmfRunTagID);

       m_runIOV.setConnection(m_env, m_conn);
       m_runIOV.setByID(runIOVID);

       m_ID = id;
     } else {
       throw(runtime_error("LMFRunIOV::setByID:  Given tag_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("LMFRunIOV::setByID:  "+e.getMessage()));
   }
}



int LMFRunIOV::writeDB()
  throw(runtime_error)
{
  this->checkConnection();

  // Check if this IOV has already been written
  if (this->fetchID()) {
    return m_ID;
  }

  // fetch Parent IDs
  int lmfRunTagID, runIOVID;
  this->fetchParentIDs(&lmfRunTagID, &runIOVID);
		       
  if (!lmfRunTagID) {
    lmfRunTagID = m_lmfRunTag.writeDB();
  }
  
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  if (m_subRunStart.isNull()) {
    throw(runtime_error("LMFRunIOV::writeDB:  Must setSubRunStart before writing"));
  }
  
  if (m_subRunEnd.isNull()) {
    m_subRunEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    
    stmt->setSQL("INSERT INTO lmf_run_iov (lmf_iov_id, tag_id, run_iov_id, subrun_num, subrun_start, subrun_end, subrun_type) "
		 "VALUES (lmf_run_iov_sq.NextVal, :1, :2, :3, :4, :5, :6)");
    stmt->setInt(1, lmfRunTagID);
    stmt->setInt(2, runIOVID);
    stmt->setInt(3, m_subRunNum);
    stmt->setDate(4, dh.tmToDate(m_subRunStart));
    stmt->setDate(5, dh.tmToDate(m_subRunEnd));
    stmt->setString(6, m_subrun_type);

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("LMFRunIOV::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("LMFRunIOV::writeDB:  Failed to write"));
  }
  
  return m_ID;
}



void LMFRunIOV::fetchParentIDs(int* lmfRunTagID, int* runIOVID)
  throw(runtime_error)
{
  // get the LMFRunTag
  m_lmfRunTag.setConnection(m_env, m_conn);
  *lmfRunTagID = m_lmfRunTag.fetchID();

  // get the RunIOV
  m_runIOV.setConnection(m_env, m_conn);
  *runIOVID = m_runIOV.fetchID();

  if (! *runIOVID) { 
    throw(runtime_error("LMFRunIOV:  Given RunIOV does not exist in DB")); 
  }

}



void LMFRunIOV::setByRun(LMFRunTag* lmftag, RunIOV* runiov, subrun_t subrun)
  throw(std::runtime_error)
{
  this->checkConnection();
  
  runiov->setConnection(m_env, m_conn);
  int runIOVID = runiov->fetchID();

  if (!runIOVID) {
    throw(runtime_error("LMFRunIOV::setByRun:  Given RunIOV does not exist in DB"));
  }

  lmftag->setConnection(m_env, m_conn);
  int lmfTagID = lmftag->fetchID();
  
  if (!lmfTagID) {
    throw(runtime_error("LMFRunIOV::setByRun:  Given LMFRunTag does not exist in the DB"));
  }

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT lmf_iov_id, subrun_start, subrun_end FROM lmf_run_iov "
		 "WHERE tag_id = :1 AND run_iov_id = :2 AND subrun_num = :3");
    stmt->setInt(1, lmfTagID);
    stmt->setInt(2, runIOVID);
    stmt->setInt(3, subrun);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_lmfRunTag = *lmftag;
      m_runIOV = *runiov;
      m_subRunNum = subrun;
      
      m_ID = rset->getInt(1);
      Date startDate = rset->getDate(2);
      Date endDate = rset->getDate(3);
	 
      m_subRunStart = dh.dateToTm( startDate );
      m_subRunEnd = dh.dateToTm( endDate );
    } else {
      throw(runtime_error("LMFRunIOV::setByRun:  Given subrun is not in the database"));
    }
     
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("LMFRunIOV::setByRun:  "+e.getMessage()));
  }
  
}
