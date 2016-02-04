#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

RunList::RunList()
{
  m_conn = NULL;
}

RunList::~RunList()
{
}

void RunList::setRunTag(RunTag tag)
{
  if (tag != m_runTag) {
    m_runTag = tag;
  }
}


RunTag RunList::getRunTag() const
{
  return m_runTag;
}

 std::vector<RunIOV> RunList::getRuns() 
{
  return m_vec_runiov;
}

void RunList::fetchRuns()
  throw(std::runtime_error)
{
  fetchRuns(-1, -1);
}

void RunList::fetchRuns(int min_run, int max_run)
  throw(std::runtime_error)
{

  this->checkConnection();
  int nruns=0;

  m_runTag.setConnection(m_env, m_conn);
  int tagID = m_runTag.fetchID();
  cout <<"tag id="<< tagID << endl;
  if (!tagID) { 
    return ;
  }

  int my_min_run=min_run-1;
  int my_max_run=max_run+1;
  try {
    Statement* stmt0 = m_conn->createStatement();
    string sql =  "SELECT count(iov_id) FROM run_iov "
      "WHERE tag_id = :tag_id ";
    if (min_run > 0) {
      sql += " and run_iov.run_num> :min_run and run_iov.run_num< :max_run ";
    }
    stmt0->setSQL(sql);
    stmt0->setInt(1, tagID);
    if (min_run > 0) {
      stmt0->setInt(2, my_min_run);
      stmt0->setInt(3, my_max_run);
    }
    
    ResultSet* rset0 = stmt0->executeQuery();
    if (rset0->next()) {
      nruns = rset0->getInt(1);
    }
    m_conn->terminateStatement(stmt0);

    cout <<"number of runs="<< nruns << endl;
    m_vec_runiov.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    sql = "SELECT iov_id, tag_id, run_num, run_start, run_end, " 
      "db_timestamp FROM run_iov "
      " WHERE tag_id = :tag_id ";
    if (min_run > 0) {
      sql += " and run_iov.run_num> :min_run and run_iov.run_num< :max_run ";
    }
    sql += 		 " order by run_num ";
    stmt->setSQL(sql);
    stmt->setInt(1, tagID);
    if (min_run > 0) {
      stmt->setInt(2, my_min_run);
      stmt->setInt(3, my_max_run);
    }

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
    Tm dbtime;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
      int iovID = rset->getInt(1);
      // int tagID = rset->getInt(2);
      int runNum = rset->getInt(3);
      Date startDate = rset->getDate(4);
      Date endDate = rset->getDate(5);
      Date dbDate = rset->getDate(6);
	 
      runStart = dh.dateToTm( startDate );
      runEnd = dh.dateToTm( endDate );
      dbtime = dh.dateToTm( dbDate );
       
      RunIOV r ;
      r.setRunNumber(runNum);
      r.setRunStart(runStart);
      r.setRunEnd(runEnd);
      r.setDBInsertionTime(dbtime);
      r.setRunTag(m_runTag);
      r.setID(iovID);
      m_vec_runiov.push_back(r);
      
      i++;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunList::fetchRuns:  "+e.getMessage()));
  }
}

void RunList::fetchLastNRuns( int max_run, int n_runs  )
  throw(std::runtime_error)
{

  // fetch the last n_runs that come just before max_run (including max_run)

  this->checkConnection();


  m_runTag.setConnection(m_env, m_conn);
  int tagID = m_runTag.fetchID();
  cout <<"tag id="<< tagID << endl;
  if (!tagID) { 
    return ;
  }

  int my_max_run=max_run+1;
  try {

    int nruns=n_runs+1;
    m_vec_runiov.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("select iov_id, tag_id, run_num, run_start, run_end, DB_TIMESTAMP from "
		 " (SELECT * from RUN_IOV "
		 " WHERE tag_id = :tag_id "
		 " and run_num< :max_run "
		 " order by run_num DESC ) where rownum< :n_runs ORDER BY run_num ASC " );
    stmt->setInt(1, tagID);
    stmt->setInt(2, my_max_run);
    stmt->setInt(3, nruns);

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
    Tm dbtime;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<n_runs) {
      rset->next();
      int iovID = rset->getInt(1);
      // int tagID = rset->getInt(2);
       int runNum = rset->getInt(3);
       Date startDate = rset->getDate(4);
       Date endDate = rset->getDate(5);
       Date dbDate = rset->getDate(6);
	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       dbtime = dh.dateToTm( dbDate );
       
       RunIOV r ;
       r.setRunNumber(runNum);
       r.setRunStart(runStart);
       r.setRunEnd(runEnd);
       r.setDBInsertionTime(dbtime);
       r.setRunTag(m_runTag);
       r.setID(iovID);
       m_vec_runiov.push_back(r);
      
      i++;
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunList::fetchLastNRuns:  "+e.getMessage()));
  }


}



void RunList::fetchRunsByLocation (int min_run, int max_run, const LocationDef locDef )
  throw(std::runtime_error)
{

  this->checkConnection();
  int nruns=0;

  int my_min_run=min_run-1;
  int my_max_run=max_run+1;
  try {
    Statement* stmt0 = m_conn->createStatement();
    stmt0->setSQL("SELECT count(iov_id) FROM run_iov r , run_tag t, location_def l "
                 " WHERE r.tag_id=t.tag_id and t.LOCATION_ID=l.def_id AND l.LOCATION= :1 " 
		  "  and r.run_num> :2 and r.run_num< :3 ");
    stmt0->setString(1,locDef.getLocation() );
    stmt0->setInt(2, my_min_run);
    stmt0->setInt(3, my_max_run);
  
    ResultSet* rset0 = stmt0->executeQuery();
    if (rset0->next()) {
      nruns = rset0->getInt(1);
    }
    m_conn->terminateStatement(stmt0);

    cout <<"number of runs="<< nruns << endl;
    
    m_vec_runiov.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT  r.iov_id, r.tag_id, r.run_num, r.run_start, r.run_end, r.DB_TIMESTAMP , "
                 " t.gen_tag, rt.RUN_TYPE "
		 " FROM run_iov r , run_tag t, location_def l, run_type_def rt "
		 " WHERE r.tag_id=t.tag_id and t.LOCATION_ID=l.def_id and t.run_type_id=rt.DEF_ID "
		 " AND l.LOCATION= :1 "
		 " and r.run_num> :2 and r.run_num< :3 "
		 " order by run_num " );
    stmt->setString(1,locDef.getLocation() );
    stmt->setInt(2, my_min_run);
    stmt->setInt(3, my_max_run);

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
    Tm dbtime;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
      int iovID = rset->getInt(1);
      //       int tagID = rset->getInt(2);
       int runNum = rset->getInt(3);
       Date startDate = rset->getDate(4);
       Date endDate = rset->getDate(5);
       Date dbDate = rset->getDate(6);
	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       dbtime = dh.dateToTm( dbDate );
       
       RunTag atag;
       atag.setLocationDef(locDef);
       atag.setGeneralTag(rset->getString(7));
       RunTypeDef rundef;
       rundef.setRunType(rset->getString(8));
       atag.setRunTypeDef(rundef);

       RunIOV r ;
       r.setRunNumber(runNum);
       r.setRunStart(runStart);
       r.setRunEnd(runEnd);
       r.setDBInsertionTime(dbtime);
       r.setRunTag(atag);
       r.setID(iovID);
       m_vec_runiov.push_back(r);
      
      i++;
    }
   

    m_conn->terminateStatement(stmt);

  } catch (SQLException &e) {
    throw(std::runtime_error("RunList::fetchRunsByLocation:  "+e.getMessage()));
  }


}

void RunList::fetchGlobalRunsByLocation (int min_run, int max_run, const LocationDef locDef )
  throw(std::runtime_error)
{

  this->checkConnection();
  int nruns=0;

  int my_min_run=min_run-1;
  int my_max_run=max_run+1;
  try {
    Statement* stmt0 = m_conn->createStatement();
    stmt0->setSQL("SELECT count(iov_id) FROM run_iov r , run_tag t, location_def l "
                 " WHERE r.tag_id=t.tag_id and t.LOCATION_ID=l.def_id AND l.LOCATION= :1 " 
		  "  and t.gen_tag='GLOBAL' "
		  "  and r.run_num> :2 and r.run_num< :3 ");
    stmt0->setString(1,locDef.getLocation() );
    stmt0->setInt(2, my_min_run);
    stmt0->setInt(3, my_max_run);
  
    ResultSet* rset0 = stmt0->executeQuery();
    if (rset0->next()) {
      nruns = rset0->getInt(1);
    }
    m_conn->terminateStatement(stmt0);

    cout <<"number of runs="<< nruns << endl;
    
    m_vec_runiov.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT  r.iov_id, r.tag_id, r.run_num, r.run_start, r.run_end, r.DB_TIMESTAMP , "
                 " t.gen_tag, rt.RUN_TYPE "
		 " FROM run_iov r , run_tag t, location_def l, run_type_def rt "
		 " WHERE r.tag_id=t.tag_id and t.LOCATION_ID=l.def_id and t.run_type_id=rt.DEF_ID "
		 " AND l.LOCATION= :1 "
		 " and t.gen_tag='GLOBAL' "
		 " and r.run_num> :2 and r.run_num< :3 "
		 " order by run_num " );
    stmt->setString(1,locDef.getLocation() );
    stmt->setInt(2, my_min_run);
    stmt->setInt(3, my_max_run);

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
    Tm dbtime;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
      int iovID = rset->getInt(1);
      //       int tagID = rset->getInt(2);
       int runNum = rset->getInt(3);
       Date startDate = rset->getDate(4);
       Date endDate = rset->getDate(5);
       Date dbDate = rset->getDate(6);
	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       dbtime = dh.dateToTm( dbDate );
       
       RunTag atag;
       atag.setLocationDef(locDef);
       atag.setGeneralTag(rset->getString(7));
       RunTypeDef rundef;
       rundef.setRunType(rset->getString(8));
       atag.setRunTypeDef(rundef);

       RunIOV r ;
       r.setRunNumber(runNum);
       r.setRunStart(runStart);
       r.setRunEnd(runEnd);
       r.setDBInsertionTime(dbtime);
       r.setRunTag(atag);
       r.setID(iovID);
       m_vec_runiov.push_back(r);
      
      i++;
    }
   

    m_conn->terminateStatement(stmt);

  } catch (SQLException &e) {
    throw(std::runtime_error("RunList::fetchRunsByLocation:  "+e.getMessage()));
  }


}
