#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonRunList.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

MonRunList::MonRunList()
{
  m_conn = NULL;
}

MonRunList::~MonRunList()
{
}

void MonRunList::setRunTag(RunTag tag)
{
  if (tag != m_runTag) {
    m_runTag = tag;
  }
}
void MonRunList::setMonRunTag(MonRunTag tag)
{
  if (tag != m_monrunTag) {
    m_monrunTag = tag;
  }
}


RunTag MonRunList::getRunTag() const
{
  return m_runTag;
}
MonRunTag MonRunList::getMonRunTag() const
{
  return m_monrunTag;
}

std::vector<MonRunIOV> MonRunList::getRuns() 
{
  return m_vec_monruniov;
}


void MonRunList::fetchRuns()
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
  m_monrunTag.setConnection(m_env, m_conn);
  int montagID = m_monrunTag.fetchID();
  cout <<"mon tag id="<< montagID << endl;
  if (!montagID) { 
    return ;
  }

  try {
    Statement* stmt0 = m_conn->createStatement();
    stmt0->setSQL("SELECT count(mon_run_iov.iov_id) FROM mon_run_iov, run_iov "
		 "WHERE mon_run_iov.run_iov_id= run_iov.iov_id and run_iov.tag_id = :tag_id and mon_run_iov.tag_id=:montag_id " );
    stmt0->setInt(1, tagID);
    stmt0->setInt(2, montagID);
  
    ResultSet* rset0 = stmt0->executeQuery();
    if (rset0->next()) {
      nruns = rset0->getInt(1);
    }
    m_conn->terminateStatement(stmt0);

    cout <<"nruns="<< nruns << endl;
    
    m_vec_monruniov.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT run_iov.run_num, run_iov.run_start, run_iov.run_end, mon_run_iov.tag_id, mon_run_iov.run_iov_id, mon_run_iov.subrun_num, mon_run_iov.subrun_start, mon_run_iov.subrun_end, mon_run_iov.iov_id FROM run_iov, mon_run_iov "
		 "WHERE mon_run_iov.run_iov_id=run_iov.iov_id and run_iov.tag_id = :tag_id  order by run_iov.run_num, mon_run_iov.subrun_num  ASC " );
    stmt->setInt(1, tagID);

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
    Tm lrunStart;
    Tm lrunEnd;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
       int runNum = rset->getInt(1);
       Date startDate = rset->getDate(2);
       Date endDate = rset->getDate(3);
       //int ltag = rset->getInt(4);
       int lid=rset->getInt(5);
       int subrun=rset->getInt(6);
       Date monstartDate = rset->getDate(7);
       Date monendDate = rset->getDate(8);
       int liov_id=rset->getInt(9);
	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       lrunStart = dh.dateToTm( monstartDate );
       lrunEnd = dh.dateToTm( monendDate );
       
       RunIOV r ;
       r.setRunNumber(runNum);
       r.setRunStart(runStart);
       r.setRunEnd(runEnd);
       r.setRunTag(m_runTag);
       r.setID(lid);
    
       MonRunIOV lr ;
       // da correggere qui
       lr.setRunIOV(r);
       lr.setSubRunNumber(subrun);
       lr.setSubRunStart(lrunStart);
       lr.setSubRunEnd(lrunEnd);
       lr.setMonRunTag(m_monrunTag);
       lr.setID(liov_id);
       m_vec_monruniov.push_back(lr);
      
      i++;
    }
   

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunIOV::fetchID:  "+e.getMessage()));
  }


}

void MonRunList::fetchRuns(int min_run, int max_run)
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
  m_monrunTag.setConnection(m_env, m_conn);
  int montagID = m_monrunTag.fetchID();
  cout <<"mon tag id="<< montagID << endl;
  if (!montagID) { 
    return ;
  }

  int my_min_run=min_run-1;
  int my_max_run=max_run+1;
  try {
    Statement* stmt0 = m_conn->createStatement();
    stmt0->setSQL("SELECT count(mon_run_iov.iov_id) FROM mon_run_iov, run_iov "
		 "WHERE mon_run_iov.run_iov_id= run_iov.iov_id "
		  "and run_iov.tag_id = :tag_id and mon_run_iov.tag_id=:montag_id "
		  " and run_iov.run_num> :min_run and run_iov.run_num< :max_run "
		  );
    stmt0->setInt(1, tagID);
    stmt0->setInt(2, montagID);
    stmt0->setInt(3, my_min_run);
    stmt0->setInt(4, my_max_run);
  
    ResultSet* rset0 = stmt0->executeQuery();
    if (rset0->next()) {
      nruns = rset0->getInt(1);
    }
    m_conn->terminateStatement(stmt0);

    cout <<"nruns="<< nruns << endl;
    
    m_vec_monruniov.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT run_iov.run_num, run_iov.run_start, run_iov.run_end, mon_run_iov.tag_id, mon_run_iov.run_iov_id, mon_run_iov.subrun_num, mon_run_iov.subrun_start, mon_run_iov.subrun_end, mon_run_iov.iov_id FROM run_iov, mon_run_iov "
		 "WHERE mon_run_iov.run_iov_id=run_iov.iov_id and run_iov.tag_id = :tag_id "
		 " and mon_run_iov.tag_id=:montag_id "
		 " and run_iov.run_num> :min_run and run_iov.run_num< :max_run "
		 " order by run_iov.run_num, mon_run_iov.subrun_num ASC " );
    stmt->setInt(1, tagID);
    stmt->setInt(2, montagID);
    stmt->setInt(3, my_min_run);
    stmt->setInt(4, my_max_run);

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
    Tm lrunStart;
    Tm lrunEnd;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
       int runNum = rset->getInt(1);
       Date startDate = rset->getDate(2);
       Date endDate = rset->getDate(3);
       // int ltag = rset->getInt(4);
       int lid=rset->getInt(5);
       int subrun=rset->getInt(6);
       Date monstartDate = rset->getDate(7);
       Date monendDate = rset->getDate(8);
       int liov_id=rset->getInt(9);
	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       lrunStart = dh.dateToTm( monstartDate );
       lrunEnd = dh.dateToTm( monendDate );
       
       RunIOV r ;
       r.setRunNumber(runNum);
       r.setRunStart(runStart);
       r.setRunEnd(runEnd);
       r.setRunTag(m_runTag);
       r.setID(lid);
    
       MonRunIOV lr ;
       // da correggere qui
       lr.setRunIOV(r);
       lr.setSubRunNumber(subrun);
       lr.setSubRunStart(lrunStart);
       lr.setSubRunEnd(lrunEnd);
       lr.setMonRunTag(m_monrunTag);
       lr.setID(liov_id);
       m_vec_monruniov.push_back(lr);
      
      i++;
    }
   

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunIOV::fetchID:  "+e.getMessage()));
  }


}

void MonRunList::fetchLastNRuns( int max_run, int n_runs  )
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
  m_monrunTag.setConnection(m_env, m_conn);
  int montagID = m_monrunTag.fetchID();
  cout <<"mon tag id="<< montagID << endl;
  if (!montagID) { 
    return ;
  }

  int my_max_run=max_run+1;
  try {

    int nruns=n_runs;
    m_vec_monruniov.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("select run_num, run_start, run_end, tag_id, run_iov_id, subrun_num, subrun_start, subrun_end, mon_iov_id from (SELECT run_iov.run_num, run_iov.run_start, run_iov.run_end, mon_run_iov.tag_id, mon_run_iov.run_iov_id, mon_run_iov.subrun_num, mon_run_iov.subrun_start, mon_run_iov.subrun_end, mon_run_iov.iov_id as mon_iov_id FROM run_iov, mon_run_iov "
		 "WHERE mon_run_iov.run_iov_id=run_iov.iov_id and run_iov.tag_id = :tag_id "
		 " and mon_run_iov.tag_id=:montag_id "
		 " and run_iov.run_num< :max_run "
		 " order by run_iov.run_num DESC ) where rownum< :n_runs order by run_num DESC " );
    stmt->setInt(1, tagID);
    stmt->setInt(2, montagID);
    stmt->setInt(3, my_max_run);
    stmt->setInt(4, n_runs);

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
    Tm lrunStart;
    Tm lrunEnd;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
       int runNum = rset->getInt(1);
       Date startDate = rset->getDate(2);
       Date endDate = rset->getDate(3);
       // int ltag = rset->getInt(4);
       int lid=rset->getInt(5);
       int subrun=rset->getInt(6);
       Date monstartDate = rset->getDate(7);
       Date monendDate = rset->getDate(8);
       int liov_id=rset->getInt(9);
	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       lrunStart = dh.dateToTm( monstartDate );
       lrunEnd = dh.dateToTm( monendDate );
       
       RunIOV r ;
       r.setRunNumber(runNum);
       r.setRunStart(runStart);
       r.setRunEnd(runEnd);
       r.setRunTag(m_runTag);
       r.setID(lid);
    
       MonRunIOV lr ;
       // da correggere qui
       lr.setRunIOV(r);
       lr.setSubRunNumber(subrun);
       lr.setSubRunStart(lrunStart);
       lr.setSubRunEnd(lrunEnd);
       lr.setMonRunTag(m_monrunTag);
       lr.setID(liov_id);
       m_vec_monruniov.push_back(lr);
      
      i++;
    }
   

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("MonRunList::fetchLastNRuns:  "+e.getMessage()));
  }


}
