#include <stdexcept>
#include "occi.h"

#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

RunIOV::RunIOV()
{
  m_conn = NULL;
  m_ID = 0;
  m_runNum = 0;
  m_runStart = Tm();
  m_runEnd = Tm();
}



RunIOV::~RunIOV()
{
}



void RunIOV::setRunNumber(run_t run)
{
  m_ID = 0;
  m_runNum = run;
}



run_t RunIOV::getRunNumber() const
{
  return m_runNum;
}



void RunIOV::setRunStart(Tm start)
{
  m_ID = 0;
  m_runStart = start;
}



Tm RunIOV::getRunStart() const
{
  return m_runStart;
}



void RunIOV::setRunEnd(Tm end)
{
  m_ID = 0;
  m_runEnd = end;
}



Tm RunIOV::getRunEnd() const
{
  return m_runEnd;
}



void RunIOV::setRunTag(RunTag tag)
{
  m_ID = 0;
  m_runTag = tag;
}



RunTag RunIOV::getRunTag() const
{
  return m_runTag;
}



int RunIOV::fetchID()
  throw(runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  m_runTag.setConnection(m_env, m_conn);
  int tagID = m_runTag.fetchID();
  if (!tagID) { 
    return 0;
  }

  DateHandler dh(m_env, m_conn);

  if (m_runEnd.isNull()) {
    m_runEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT iov_id FROM run_iov "
		 "WHERE tag_id = :tag_id AND "
		 "run_num = :run_num AND "
		 "run_start = :run_start AND "
		 "run_end = :run_end");
    stmt->setInt(1, tagID);
    stmt->setInt(2, m_runNum);
    stmt->setDate(3, dh.tmToDate(m_runStart));
    stmt->setDate(4, dh.tmToDate(m_runEnd));
  
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("RunIOV::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void RunIOV::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT tag_id, run_num, run_start, run_end FROM run_iov WHERE iov_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       int tagID = rset->getInt(1);
       m_runNum = rset->getInt(2);
       Date startDate = rset->getDate(3);
       Date endDate = rset->getDate(4);
	 
       m_runStart = dh.dateToTm( startDate );
       m_runEnd = dh.dateToTm( endDate );

       m_runTag.setConnection(m_env, m_conn);
       m_runTag.setByID(tagID);
       m_ID = id;
     } else {
       throw(runtime_error("RunTag::setByID:  Given tag_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("RunTag::setByID:  "+e.getMessage()));
   }
}



int RunIOV::writeDB()
  throw(runtime_error)
{
  this->checkConnection();

  // Check if this IOV has already been written
  if (this->fetchID()) {
    return m_ID;
  }

  m_runTag.setConnection(m_env, m_conn);
  int tagID = m_runTag.writeDB();
  
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  if (m_runStart.isNull()) {
    throw(runtime_error("RunIOV::writeDB:  Must setRunStart before writing"));
  }
  
  if (m_runEnd.isNull()) {
    m_runEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    
    stmt->setSQL("INSERT INTO run_iov (iov_id, tag_id, run_num, run_start, run_end) "
		 "VALUES (run_iov_sq.NextVal, :1, :2, :3, :4)");
    stmt->setInt(1, tagID);
    stmt->setInt(2, m_runNum);
    stmt->setDate(3, dh.tmToDate(m_runStart));
    stmt->setDate(4, dh.tmToDate(m_runEnd));

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("RunIOV::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("RunIOV::writeDB:  Failed to write"));
  }
  
  return m_ID;
}



void RunIOV::setByRun(RunTag* tag, run_t run) 
  throw(std::runtime_error)
{
   this->checkConnection();

   tag->setConnection(m_env, m_conn);
   int tagID = tag->fetchID();
   if (!tagID) {
     throw(runtime_error("RunIOV::setByRun:  Given tag is not in the database"));
   }
   
   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT iov_id, run_start, run_end FROM run_iov WHERE tag_id = :1 AND run_num = :2");
     stmt->setInt(1, tagID);
     stmt->setInt(2, run);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_runTag = *tag;
       m_runNum = run;

       m_ID = rset->getInt(1);
       Date startDate = rset->getDate(2);
       Date endDate = rset->getDate(3);
	 
       m_runStart = dh.dateToTm( startDate );
       m_runEnd = dh.dateToTm( endDate );
     } else {
       throw(runtime_error("RunIOV::setByRun:  Given run is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("RunIOV::setByRun:  "+e.getMessage()));
   }
}




// void RunIOV::fetchEarliest(RunIOV* fillIOV, RunTag* tag) const
//   throw(runtime_error)
// {
//   string sql = "SELECT run_num, run_start, run_end FROM run_iov "
//     "WHERE run_num = (SELECT min(run_num) FROM run_iov WHERE tag_id = :tag_id)";
//   try {
//     Statement* stmt = this->prepareFetch(sql, tag);

//     ResultSet* rset = stmt->executeQuery();
  
//     if (rset->next()) {
//       this->fill(fillIOV, rset);
//     } else {
//       fillIOV = NULL;
//     }
//     m_conn->terminateStatement(stmt);
//   } catch (SQLException &e) {
//     throw(runtime_error("RunIOV::fetchEarliest():  "+e.getMessage()));
//   }
// }



// void RunIOV::fetchLatest(RunIOV* fillIOV, RunTag* tag) const
//   throw(runtime_error)
// {
//   string sql = "SELECT run_num, run_start, run_end FROM run_iov "
//     "WHERE run_num = (SELECT max(run_num) FROM run_iov WHERE tag_id = :tag_id)";
//   try {
//     Statement* stmt = this->prepareFetch(sql, tag);
//     ResultSet* rset = stmt->executeQuery();
    
//     if (rset->next()) {
//       this->fill(fillIOV, rset);
//     } else {
//       fillIOV = NULL;
//     }
//     m_conn->terminateStatement(stmt);  
//   } catch (SQLException &e) {
//     throw(runtime_error("RunIOV::fetchLatest():  "+e.getMessage()));
//   }
// }



// void RunIOV::fetchAt(RunIOV* fillIOV, const Tm eventTm, RunTag* tag) const
//   throw(std::runtime_error)
// {
//   string sql = "SELECT run_num, run_start, run_end FROM run_iov "
//     "WHERE tag_id = :tag_id AND run_start <= :t AND run_end > :t";
//   try {
//     Statement* stmt = this->prepareFetch(sql, tag);
  
//     DateHandler dh(m_env, m_conn);

//     stmt->setDate(2, dh.tmToDate(eventTm) );
//     stmt->setDate(3, dh.tmToDate(eventTm) );

//     ResultSet* rset = stmt->executeQuery();

//     if (rset->next()) {
//       this->fill(fillIOV, rset);
//     } else {
//       fillIOV = NULL;
//     }
//     m_conn->terminateStatement(stmt);
//   } catch (SQLException &e) {
//     throw(runtime_error("RunIOV::fetchAt():  "+e.getMessage()));
//   }
// }



// void RunIOV::fetchWithin(vector<RunIOV>* fillVec, const Tm beginTm, const Tm endTm, RunTag* tag) const
//   throw(std::runtime_error)
// {
//   string sql = "SELECT run_num, run_start, run_end FROM run_iov "
//     "WHERE tag_id = :tag_id AND run_start >= :t1 AND run_start <= :t2";
//   try {
//     Statement* stmt = this->prepareFetch(sql, tag);
  
//     DateHandler dh(m_env, m_conn);

//     Date beginDate = dh.tmToDate(beginTm);
//     Date endDate = dh.tmToDate(endTm);

//     stmt->setDate(2, beginDate );
//     stmt->setDate(3, endDate );

//     ResultSet* rset = stmt->executeQuery();

//     while (rset->next()) {
//       RunIOV runiov;
//       this->fill(&runiov, rset);
//       fillVec->push_back(runiov);
//     }
//     m_conn->terminateStatement(stmt);
//   } catch (SQLException &e) {
//     throw(runtime_error("RunIOV::fetchAt():  "+e.getMessage()));
//   }
// }



// void RunIOV::fetchAt(RunIOV* fillIOV, const run_t run, RunTag* tag) const
//   throw(std::runtime_error)
// {
//   string sql = "SELECT run_num, run_start, run_end FROM run_iov "
//     "WHERE tag_id = :tag_id AND run_num = :run_num";
//   try {
//     Statement* stmt = this->prepareFetch(sql, tag);
  
//     stmt->setInt(2, run);

//     ResultSet* rset = stmt->executeQuery();

//     if (rset->next()) {
//       this->fill(fillIOV, rset);
//     } else {
//       fillIOV = NULL;
//     }
//     m_conn->terminateStatement(stmt);
//   } catch (SQLException &e) {
//     throw(runtime_error("RunIOV::fetchAt():  "+e.getMessage()));
//   }
// }

// void RunIOV::fetchWithin(vector<RunIOV>* fillVec, const run_t beginRun, const run_t endRun, RunTag* tag) const
//   throw(std::runtime_error)
// {
//   string sql = "SELECT run_num, run_start, run_end FROM run_iov "
//     "WHERE tag_id = :tag_id AND run_num >= :r1 AND run_num <= :r2";
//   try {
//     Statement* stmt = this->prepareFetch(sql, tag);

//     stmt->setInt(2, beginRun);
//     stmt->setInt(3, endRun);

//     ResultSet* rset = stmt->executeQuery();
    
//     while (rset->next()) {
//       RunIOV runiov;
//       this->fill(&runiov, rset);
//       fillVec->push_back(runiov);
//     }
//     m_conn->terminateStatement(stmt);
//   } catch (SQLException &e) {
//     throw(runtime_error("RunIOV::fetchAt():  "+e.getMessage()));
//   }
// }



// Statement* RunIOV::prepareFetch(const string sql, RunTag* tag) const
//   throw(runtime_error)
// {
//   this->checkConnection();

//   tag->setConnection(m_env, m_conn);
//   int tagID = tag->fetchID();
//   if (!tagID) { 
//     throw(runtime_error("RunIOV::prepareFetch():  Given tag does not exist in DB")); 
//   }
  
//   Statement* stmt;

//   try {
//     stmt = m_conn->createStatement();
//     stmt->setSQL(sql);
//     stmt->setInt(1, tagID);
//   }  catch (SQLException &e) {
//     throw(runtime_error("RunIOV::prepareFetch():  "+e.getMessage()));
//   }
  
//   return stmt;
// }

// void RunIOV::fill(RunIOV* iov, ResultSet* rset) const
//   throw(runtime_error)
// {
//   DateHandler dh(m_env, m_conn);

//   try {
//     iov->setRunNumber( rset->getInt(1) );
//     Date startDate = rset->getDate(2);
//     Tm startTm = dh.dateToTm(startDate);

//     iov->setRunStart( startTm );
//     Date endDate = rset->getDate(3);
//     Tm endTm = dh.dateToTm(endDate);

//     iov->setRunEnd( endTm );
//   } catch(SQLException &e) {
//     throw(runtime_error("RunIOV::fill:  "+e.getMessage()));
//   }
  
// }
