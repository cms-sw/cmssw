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
  throw(runtime_error)
{


  this->checkConnection();
  int nruns=0;

  m_runTag.setConnection(m_env, m_conn);
  int tagID = m_runTag.fetchID();
  cout <<"tag id="<< tagID << endl;
  if (!tagID) { 
    return ;
  }

  try {
    Statement* stmt0 = m_conn->createStatement();
    stmt0->setSQL("SELECT count(iov_id) FROM run_iov "
		 "WHERE tag_id = :tag_id  " );
    stmt0->setInt(1, tagID);
  
    ResultSet* rset0 = stmt0->executeQuery();
    if (rset0->next()) {
      nruns = rset0->getInt(1);
    }
    m_conn->terminateStatement(stmt0);

    cout <<"nruns="<< nruns << endl;
    m_vec_runiov.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT iov_id, tag_id, run_num, run_start, run_end FROM run_iov "
		 "WHERE tag_id = :tag_id  order by run_num " );
    stmt->setInt(1, tagID);

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
      int iovID = rset->getInt(1);
       int tagID = rset->getInt(2);
       int runNum = rset->getInt(3);
       Date startDate = rset->getDate(4);
       Date endDate = rset->getDate(5);
	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       
       RunIOV r ;
       r.setRunNumber(runNum);
       r.setRunStart(runStart);
       r.setRunEnd(runEnd);
       r.setRunTag(m_runTag);
       r.setID(iovID);
       m_vec_runiov.push_back(r);
      
      i++;
    }
   

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("RunList::fetchRuns:  "+e.getMessage()));
  }


}

