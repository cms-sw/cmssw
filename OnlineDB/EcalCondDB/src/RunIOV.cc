#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

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
  if ( run != m_runNum) {
    m_ID = 0;
    m_runNum = run;
  }
}

void RunIOV::setID(int id)
{
     m_ID = id;
   }




run_t RunIOV::getRunNumber() const
{
  return m_runNum;
}



void RunIOV::setRunStart(Tm start)
{
  if (start != m_runStart) {
    m_ID = 0;
    m_runStart = start;
  }
}



Tm RunIOV::getRunStart() const
{
  return m_runStart;
}



void RunIOV::setRunEnd(Tm end)
{
  if (end != m_runEnd) {
    m_ID = 0;
    m_runEnd = end;
  }
}



Tm RunIOV::getRunEnd() const
{
  return m_runEnd;
}



void RunIOV::setRunTag(RunTag tag)
{
  if (tag != m_runTag) {
    m_ID = 0;
    m_runTag = tag;
  }
}



RunTag RunIOV::getRunTag() const
{
  return m_runTag;
}



int RunIOV::fetchID()
  throw(std::runtime_error)
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
		 "run_start = :run_start  " );
    stmt->setInt(1, tagID);
    stmt->setInt(2, m_runNum);
    stmt->setDate(3, dh.tmToDate(m_runStart));
  
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunIOV::fetchID:  "+e.getMessage()));
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
       throw(std::runtime_error("RunIOV::setByID:  Given tag_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("RunIOV::setByID:  "+e.getMessage()));
   }
}



int RunIOV::writeDB()
  throw(std::runtime_error)
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
    throw(std::runtime_error("RunIOV::writeDB:  Must setRunStart before writing"));
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
    throw(std::runtime_error("RunIOV::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("RunIOV::writeDB:  Failed to write"));
  }
  
  return m_ID;
}


int RunIOV::updateEndTimeDB()
  throw(std::runtime_error)
{
  this->checkConnection();

  // Check if this IOV has already been written
  if(!this->fetchID()){
    this->writeDB();
  }


  m_runTag.setConnection(m_env, m_conn);
  //  int tagID = m_runTag.writeDB();
  
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  // we only update the run end here   
  if (m_runEnd.isNull()) {
    m_runEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    
    stmt->setSQL("UPDATE run_iov set run_end=:1 where iov_id=:2 " );
    stmt->setDate(1, dh.tmToDate(m_runEnd));
    stmt->setInt(2, m_ID);

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunIOV::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("RunIOV::writeDB:  Failed to write"));
  }
  
  return m_ID;
}

int RunIOV::fetchIDByRunAndTag()
  throw(std::runtime_error)
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
		 "run_num = :run_num " );
    stmt->setInt(1, tagID);
    stmt->setInt(2, m_runNum);
  
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunIOV::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}


int RunIOV::updateStartTimeDB()
  throw(std::runtime_error)
{
  this->checkConnection();

  // Check if this IOV has already been written
  if(!this->fetchIDByRunAndTag()){
    this->writeDB();
  }


  //  m_runTag.setConnection(m_env, m_conn);
  // int tagID = m_runTag.writeDB();
  
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  // we only update the run start here   
  if (m_runEnd.isNull()) {
    m_runEnd = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    
    stmt->setSQL("UPDATE run_iov set run_start=:1 where iov_id=:2 " );
    stmt->setDate(1, dh.tmToDate(m_runStart));
    stmt->setInt(2, m_ID);

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunIOV::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("RunIOV::writeDB:  Failed to write"));
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
     throw(std::runtime_error("RunIOV::setByRun:  Given tag is not in the database"));
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
       throw(std::runtime_error("RunIOV::setByRun:  Given run is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("RunIOV::setByRun:  "+e.getMessage()));
   }
}



void RunIOV::setByRun(std::string location, run_t run) 
  throw(std::runtime_error)
{
  this->checkConnection();
   
  DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT iov_id FROM run_iov riov "
		  "JOIN run_tag rtag ON riov.tag_id = rtag.tag_id "
		  "JOIN location_def loc ON rtag.location_id = loc.def_id "
		  "WHERE loc.location = :1 AND riov.run_num = :2 "
		  "AND rtag.gen_tag != 'INVALID'");
     stmt->setString(1, location);
     stmt->setInt(2, run);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       int id = rset->getInt(1);
       this->setByID(id);
     } else {
       throw(std::runtime_error("RunIOV::setByRun(loc, run):  Given run is not in the database"));
     }
     
     // Check for uniqueness of run
     if (rset->next()) {
       throw(std::runtime_error("RunIOV::setByRun(loc, run):  Run is nonunique for given location."));
     }

     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("RunIOV::setByRun(loc, run):  "+e.getMessage()));
   }
}


void RunIOV::setByRecentData(std::string dataTable, RunTag* tag, run_t run) 
  throw(std::runtime_error)
{
   this->checkConnection();

   tag->setConnection(m_env, m_conn);
   int tagID = tag->fetchID();
   if (!tagID) {
     throw(std::runtime_error("RunIOV::setByRecentData:  Given tag is not in the database"));
   }
   
   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT * FROM (SELECT riov.iov_id, riov.run_num, riov.run_start, riov.run_end "
		  "FROM run_iov riov "
		  "JOIN "+dataTable+" dat on dat.iov_id = riov.iov_id "
		  "WHERE tag_id = :1 AND riov.run_num <= :run ORDER BY riov.run_num DESC) WHERE rownum = 1");

     stmt->setInt(1, tagID);
     stmt->setInt(2, run);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_runTag = *tag;

       m_ID = rset->getInt(1);
       m_runNum = rset->getInt(2);
       Date startDate = rset->getDate(3);
       Date endDate = rset->getDate(4);
	 
       m_runStart = dh.dateToTm( startDate );
       m_runEnd = dh.dateToTm( endDate );
     } else {
       throw(std::runtime_error("RunIOV::setByRecentData:  No data exists for given tag and run"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("RunIOV::setByRecentData:  "+e.getMessage()));
   }
}






void RunIOV::setByRecentData(std::string dataTable, std::string location, run_t run) 
  throw(std::runtime_error)
{
  this->checkConnection();
   
  DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT * FROM (SELECT riov.iov_id, riov.run_num, riov.run_start, riov.run_end "
		  "FROM run_iov riov "
		  "JOIN "+dataTable+" dat on dat.iov_id = riov.iov_id "
	          "JOIN run_tag rtag ON riov.tag_id = rtag.tag_id "
		  "JOIN location_def loc ON rtag.location_id = loc.def_id "
		  "WHERE loc.location = :1 AND riov.run_num <= :2 ORDER BY riov.run_num DESC ) WHERE rownum = 1");

     stmt->setString(1, location);
     stmt->setInt(2, run);
     
     ResultSet* rset = stmt->executeQuery();
    

     if (rset->next()) {
       int id = rset->getInt(1);
       this->setByID(id);
     } else {
       throw(std::runtime_error("RunIOV::setByRecentData(datatable, loc, run):  Given run is not in the database"));
     }

     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("RunIOV::setByRecentData:  "+e.getMessage()));
   }
}

