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
  m_conn = nullptr;
}

RunList::~RunList()
{
}

void RunList::setRunTag(const RunTag& tag)
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

void RunList::fetchNonEmptyRuns() 
  noexcept(false)
{
  fetchRuns(-1, -1, true, false); 
}

void RunList::fetchNonEmptyGlobalRuns() 
  noexcept(false)
{
  fetchRuns(-1, -1, false, true); 
}

void RunList::fetchNonEmptyRuns(int min_run, int max_run) 
  noexcept(false)
{
  fetchRuns(min_run, max_run, true, false); 
}

void RunList::fetchNonEmptyGlobalRuns(int min_run, int max_run) 
  noexcept(false)
{
  fetchRuns(min_run, max_run, false, true); 
}

void RunList::fetchRuns()
  noexcept(false)
{
  fetchRuns(-1, -1);
}

void RunList::fetchRuns(int min_run, int max_run)
  noexcept(false)
{
  fetchRuns(min_run, max_run, false, false);
}

void RunList::fetchRuns(int min_run, int max_run, bool withTriggers,
			bool withGlobalTriggers)
  noexcept(false)
{

  /*
    withTriggers and withGlobalTriggers selects those non empty runs.
    Possible combinations are

    withTriggers withGlobalTriggers select
    ------------ ------------------ ------------------------------
    false        false              all
    false        true               only runs with global triggers
    true         false              only runs with any trigger
    true         true               only runs with global triggers
   */
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
      // don't need to specify empty/non empty here. This is needed
      // just to allocate the memory for the vector
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
    sql = "SELECT DISTINCT i.iov_id, tag_id, run_num, run_start, run_end, " 
      "db_timestamp FROM run_iov i ";
    if ((withTriggers) || (withGlobalTriggers)) {
      sql += "join cms_ecal_cond.run_dat d on d.iov_id = i.iov_id " 
	"left join CMS_WBM.RUNSUMMARY R on R.RUNNUMBER = i.RUN_NUM ";
    }
    sql +=  "WHERE tag_id = :tag_id ";
    if (min_run > 0) {
      sql += "and i.run_num> :min_run and i.run_num< :max_run ";
    }
    if (withGlobalTriggers) {
      sql += "and R.TRIGGERS > 0 ";
    } else if (withTriggers) {
      sql += "and ((R.TRIGGERS > 0) or (num_events > 0)) ";
    }
    sql += " order by run_num ";
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
    while ((i<nruns) && (rset->next()))  {
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
    m_vec_runiov.resize(i);
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("RunList::fetchRuns:  ")+getOraMessage(&e)));
  }
}

void RunList::fetchLastNRuns( int max_run, int n_runs  )
  noexcept(false)
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
    throw(std::runtime_error(std::string("RunList::fetchLastNRuns:  ")+getOraMessage(&e)));
  }


}



void RunList::fetchRunsByLocation (int min_run, int max_run, const LocationDef& locDef )
  noexcept(false)
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
       atag.setGeneralTag(getOraString(rset,7));
       RunTypeDef rundef;
       rundef.setRunType(getOraString(rset,8));
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
    throw(std::runtime_error(std::string("RunList::fetchRunsByLocation:  ")+getOraMessage(&e)));
  }


}

void RunList::fetchGlobalRunsByLocation (int min_run, int max_run, const LocationDef& locDef )
  noexcept(false)
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
       atag.setGeneralTag(getOraString(rset,7));
       RunTypeDef rundef;
       rundef.setRunType(getOraString(rset,8));
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
    throw(std::runtime_error(std::string("RunList::fetchRunsByLocation:  ")+getOraMessage(&e)));
  }


}
