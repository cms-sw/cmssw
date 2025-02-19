#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"
#include "OnlineDB/EcalCondDB/interface/DCSPTMTemp.h"
#include "OnlineDB/EcalCondDB/interface/DCSPTMTempList.h"
#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

using namespace std;
using namespace oracle::occi;


DCSPTMTempList::DCSPTMTempList()
{
  m_conn = NULL;
}

DCSPTMTempList::~DCSPTMTempList()
{
}


 std::vector<DCSPTMTemp> DCSPTMTempList::getList() 
{
  return m_vec_temp;
}


void DCSPTMTempList::fetchValuesForECID(EcalLogicID ecid)
  throw(std::runtime_error)
{

  this->checkConnection();
  int nruns=0;

  int ecid_id=ecid.getLogicID();

  try {
    Statement* stmt0 = m_conn->createStatement();
    stmt0->setSQL("SELECT count(since) FROM PVSS_TEMPERATURE_DAT "
		 "WHERE logic_id = :logic_id  " );
    stmt0->setInt(1, ecid_id);
  
    ResultSet* rset0 = stmt0->executeQuery();
    if (rset0->next()) {
      nruns = rset0->getInt(1);
    }
    m_conn->terminateStatement(stmt0);

    cout <<"DCSPTMTempList::fetchValuesForECID>> Number of records in DB="<< nruns << endl;
    m_vec_temp.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT  "
		 "since, till, temperature  FROM PVSS_TEMPERATURE_DAT "
		 "WHERE logic_id = :logic_id  order by since " );
    stmt->setInt(1, ecid_id);

    DateHandler dh(m_env, m_conn);
    Tm runStart;
    Tm runEnd;
  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
     
       Date startDate = rset->getDate(1);
       Date endDate = rset->getDate(2);	 
       float x = rset->getFloat(3);	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       
       DCSPTMTemp r ;
       r.setTemperature(x);
       r.setStart(runStart);
       r.setEnd(runEnd);
       r.setEcalLogicID(ecid);
       m_vec_temp.push_back(r);
      
      i++;
    }
   
    cout <<"DCSPTMTempList::fetchValuesForECID>> loop done " << endl;

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("DCSPTMTempList:  "+e.getMessage()));
  }


}

void DCSPTMTempList::fetchValuesForECIDAndTime(EcalLogicID ecid, Tm start, Tm end)
  throw(std::runtime_error)
{

  this->checkConnection();
  int nruns=0;

  int ecid_id=ecid.getLogicID();

  DateHandler dh(m_env, m_conn);
  Tm runStart;
  Tm runEnd;


  try {
    Statement* stmt0 = m_conn->createStatement();
    stmt0->setSQL("SELECT count(since) FROM PVSS_TEMPERATURE_DAT "
		 "WHERE logic_id = :logic_id  " 
		 "AND since >= :start_time  " 
		 "AND since <= :till_time  " 
		  );
    stmt0->setInt(1, ecid_id);
    stmt0->setDate(2, dh.tmToDate(start));
    stmt0->setDate(3, dh.tmToDate(end));
  
    ResultSet* rset0 = stmt0->executeQuery();
    if (rset0->next()) {
      nruns = rset0->getInt(1);
    }
    m_conn->terminateStatement(stmt0);

    cout <<"DCSPTMTempList::fetchValuesForECIDAndTime>> Number of records in DB="<< nruns << endl;
    m_vec_temp.reserve(nruns);
    
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT  "
		 "since, till, temperature  FROM PVSS_TEMPERATURE_DAT "
		 "WHERE logic_id = :logic_id "
		 "AND since >= :start_time  " 
		 "AND since <= :till_time  " 
		 " order by since " );
    stmt->setInt(1, ecid_id);
    stmt->setDate(2, dh.tmToDate(start));
    stmt->setDate(3, dh.tmToDate(end));

  
    ResultSet* rset = stmt->executeQuery();
    int i=0;
    while (i<nruns) {
      rset->next();
     
       Date startDate = rset->getDate(1);
       Date endDate = rset->getDate(2);	 
       float x = rset->getFloat(3);	 
       runStart = dh.dateToTm( startDate );
       runEnd = dh.dateToTm( endDate );
       
       DCSPTMTemp r ;
       r.setTemperature(x);
       r.setStart(runStart);
       r.setEnd(runEnd);
       r.setEcalLogicID(ecid);
       m_vec_temp.push_back(r);
      
      i++;
    }
   

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("DCSPTMTempList:  "+e.getMessage()));
  }


}

