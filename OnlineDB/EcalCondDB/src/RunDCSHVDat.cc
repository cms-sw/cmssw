#include <stdexcept>
#include <string>
#include <math.h>
#include <list>
#include <string>
#include <map>

#include "OnlineDB/EcalCondDB/interface/RunDCSHVDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

using namespace std;
using namespace oracle::occi;

RunDCSHVDat::RunDCSHVDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_hv = 0;
  m_hvnom = 0;
  m_status= 0;
  m_tstatus= 0;
}



RunDCSHVDat::~RunDCSHVDat()
{
}



void RunDCSHVDat::prepareWrite()
  throw(std::runtime_error)
{


}



void RunDCSHVDat::writeDB(const EcalLogicID* ecid, const RunDCSHVDat* item, RunIOV* iov)
  throw(std::runtime_error)
{
}



void RunDCSHVDat::fetchData(map< EcalLogicID, RunDCSHVDat >* fillMap, RunIOV* iov)
  throw(std::runtime_error)
{
  fetchLastData(fillMap);

}



ResultSet *RunDCSHVDat::getBarrelRset(Tm timeStart) {

  DateHandler dh(m_env, m_conn);

  ResultSet* rset = NULL;
  string query="SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, " 
    " d.actual_vmon, h.nominal_value ,  d.change_date " 
    " FROM "+ getEBAccount()+".FWCAENCHANNEL d " 
    " JOIN "+ getEBAccount()+".HV_MAPPING h on h.DPID = d.DPID " 
    " join "+ getEBAccount()+".CHANNELVIEW cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name "
    " AND d.change_date> :1 AND d.actual_vmon is not null order by change_date " ;
  try {
    m_readStmt->setSQL(query);

    m_readStmt->setDate(1, dh.tmToDate(timeStart));

    rset = m_readStmt->executeQuery();
  }
  catch (SQLException e) {
    throw(std::runtime_error("RunDCSHVDat::getBarrelRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}

ResultSet *RunDCSHVDat::getEndcapAnodeRset(Tm timeStart) {

  DateHandler dh(m_env, m_conn);

  ResultSet* rset = NULL;
  string query="SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, " 
    " d.actual_vmon, '800' nominal_value ,  d.change_date " 
    " FROM "+ getEEAccount()+".FWCAENCHANNEL d " 
    " JOIN "+ getEEAccount()+".EE_HVA_MAPPING h on h.DPID = d.DPID " 
    " join "+ getEEAccount()+".CHANNELVIEW cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name "
    " AND d.change_date> :1 AND d.actual_vmon is not null order by change_date " ;
  try {
    m_readStmt->setSQL(query);

    m_readStmt->setDate(1, dh.tmToDate(timeStart));

    rset = m_readStmt->executeQuery();
  }
  catch (SQLException e) {
    throw(std::runtime_error("RunDCSHVDat::getBarrelRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}

ResultSet *RunDCSHVDat::getEndcapDynodeRset(Tm timeStart) {

  DateHandler dh(m_env, m_conn);

  ResultSet* rset = NULL;
  string query="SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, " 
    " d.actual_vmon, '600' nominal_value ,  d.change_date " 
    " FROM "+ getEEAccount()+".FWCAENCHANNEL d " 
    " JOIN "+ getEEAccount()+".EE_HVD_MAPPING h on h.DPID = d.DPID " 
    " join "+ getEEAccount()+".CHANNELVIEW cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name "
    " AND d.change_date> :1 AND d.actual_vmon is not null order by change_date " ;
  try {
    m_readStmt->setSQL(query);

    m_readStmt->setDate(1, dh.tmToDate(timeStart));

    rset = m_readStmt->executeQuery();
  }
  catch (SQLException e) {
    throw(std::runtime_error("RunDCSHVDat::getBarrelRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}


/*
ResultSet *RunDCSHVDat::getBarrelRset() {
  ResultSet* rset = NULL;
  string query = "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
    " d.value, h.nominal_value , d.since "
    "FROM "+ getEBAccount()+".DCSLASTVALUE_VOLTAGE_VMON d "
    " JOIN "+ getEBAccount()+".HV_MAPPING h on "
    " h.DPID = d.DPID join channelview cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name"; 
  try {
    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  }
  catch (SQLException e) {
    throw(std::runtime_error("RunDCSHVDat::getBarrelRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}
*/

ResultSet *RunDCSHVDat::getBarrelRset() {
  ResultSet* rset = NULL;
  string query = "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
    " d.VALUE_NUMBER, h.nominal_value , d.CHANGE_DATE "
    "FROM "+ getEBAccount()+".FWCAENCHANNEL_LV d "
    " JOIN "+ getEBAccount()+".HV_MAPPING h on "
    " h.DPID = d.DPID join channelview cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name and DPE_NAME='ACTUAL_VMON' "; 
  try {
    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  }
  catch (SQLException e) {
    throw(std::runtime_error("RunDCSHVDat::getBarrelRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}

ResultSet *RunDCSHVDat::getEndcapAnodeRset() {
  ResultSet* rset = NULL;
  string query = "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
    " d.value_number, '800' NOMINAL_VALUE , d.CHANGE_DATE "
    "FROM "+ getEEAccount()+".FWCAENCHANNEL_LV d "
    " JOIN "+ getEEAccount()+".EE_HVA_MAPPING h on "
    " h.DPID = d.DPID join channelview cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name and dpe_name='ACTUAL_VMON' "; 
  try {
    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  }
  catch (SQLException e) {
    throw(std::runtime_error("RunDCSHVDat::getEndcapAnodeRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}

ResultSet *RunDCSHVDat::getEndcapDynodeRset() {
  ResultSet* rset = NULL;
  string query = "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
    " d.value_number, '600' NOMINAL_VALUE , d.CHANGE_DATE "
    "FROM "+ getEEAccount()+".FWCAENCHANNEL_LV d "
    " JOIN "+ getEEAccount()+".EE_HVD_MAPPING h on "
    " h.DPID = d.DPID join channelview cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name and dpe_name='ACTUAL_VMON' ";
  try {
    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  } 
  catch (SQLException e) {
    throw(std::runtime_error("RunDCSHVDat::getEndcapDynodeRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}

void RunDCSHVDat::fillTheMap(ResultSet *rset, 
			       map< EcalLogicID, RunDCSHVDat >* fillMap) {

  // method for last value queries 

  std::pair< EcalLogicID, RunDCSHVDat > p;
  RunDCSHVDat dat;
  DateHandler dh(m_env, m_conn);

  try {
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to
      
      dat.setHV(        rset->getFloat(7) );
      dat.setHVNominal( rset->getFloat(8) );
      Date sinceDate = rset->getDate(9);
      Tm  sinceTm = dh.dateToTm( sinceDate );
      dat.setStatus(0);
      if (p.first.getName() == "EB_HV_channel") {
	setStatusForBarrel(dat, sinceTm);      
      } else {
	setStatusForEndcaps(dat, sinceTm);      
      }
      p.second = dat;
      fillMap->insert(p);
    } 
  }
  catch (SQLException &e) {
    throw(std::runtime_error("RunDCSHVDat::fetchData():  "+e.getMessage()));
  }
}




void RunDCSHVDat::fillTheMapByTime(ResultSet *rset,
				   std::list<  DataReducer<RunDCSHVDat>::MyData<RunDCSHVDat>  >* my_data_list ) {
 
  //			       std::list< std::pair< Tm, std::map< EcalLogicID, RunDCSHVDat > > >* fillMap) {

  // method for historic queries

  RunDCSHVDat dat;
  DateHandler dh(m_env, m_conn);


  try {
    int count=-1;
    while(rset->next()) {
      EcalLogicID ec = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to
      

      dat.setHV(        rset->getFloat(7) );
      dat.setHVNominal( rset->getFloat(8) );
      //      Date sinceDate = rset->getDate(9);
      //     Tm  sinceTm = dh.dateToTm( sinceDate );
      // Date sinceDate = rset->getDate(9);
      Timestamp  ora_timestamp = rset->getTimestamp(9);
      Tm sinceTm; // YYYY-MM-DD HH:MM:SS
      sinceTm.setToString(ora_timestamp.toText("yyyy-mm-dd hh24:mi:ss",0));

      dat.setStatus(0);
      if (ec.getName() == "EB_HV_channel") {
	setStatusForBarrel(dat, sinceTm);      
      } else {
	setStatusForEndcaps(dat, sinceTm);      
      }

      std::pair< EcalLogicID, RunDCSHVDat > d;
      d.first=ec;
      d.second=dat;
      std::pair< Tm, std::pair< EcalLogicID, RunDCSHVDat > > p;
      p.first=sinceTm; 
      p.second = d;

      DataReducer<RunDCSHVDat>::MyData<RunDCSHVDat> pp;
      pp.m_iData=p;

      my_data_list->push_back(pp);
      count++;
      if(count<100) std::cout<<"DCS DB : size:"<< my_data_list->size()<<" Tm " <<sinceTm.str()<<" "<<ec.getID1()<<" "<<ec.getID2()<<" "<<dat.getHV()<<std::endl;
    }
      std::cout<<"DCS DB : size:"<< my_data_list->size()<<std::endl;


  }
  catch (SQLException &e) {
    throw(std::runtime_error("RunDCSHVDat::fetchData():  "+e.getMessage()));
  }
}

int  RunDCSHVDat::nowMicroseconds() {
  Tm t_now_gmt;
  
  t_now_gmt.setToCurrentGMTime();
  int t_now_gmt_micros = t_now_gmt.microsTime();
  return t_now_gmt_micros;
}



void RunDCSHVDat::setStatusForBarrel(RunDCSHVDat &dat, Tm sinceTm) {
  int t_now_gmt_micros = nowMicroseconds();

  float hv_diff=dat.getHV() - dat.getHVNominal();
  if(hv_diff<0) hv_diff=-hv_diff; 
  if (hv_diff*1000 > maxHVDifferenceEB) {
    dat.setStatus(HVNOTNOMINAL);
  }
  if (dat.getHV()*1000 < minHV) {
    dat.setStatus(HVOFF);
  }

  int result=0;
  long long d= (t_now_gmt_micros - sinceTm.microsTime()) ;
  if (d> maxDifference) {
    result= -d/1000000 ;
  } 
  dat.setTimeStatus(result);


}

void  RunDCSHVDat::setStatusForEndcaps(RunDCSHVDat &dat, Tm sinceTm) {
  int t_now_gmt_micros = nowMicroseconds();

  if (fabs(dat.getHV() - dat.getHVNominal())*1000 > maxHVDifferenceEE) {
    dat.setStatus(HVNOTNOMINAL);
  }
  if (dat.getHV()*1000 < minHV) {
    dat.setStatus(HVOFF);
  }

  int result=0;
  long long d= (t_now_gmt_micros - sinceTm.microsTime()) ;
  if (d> maxDifference) {
    result= -d/1000000 ;
  } 
  dat.setTimeStatus(result);
}

void RunDCSHVDat::fetchLastData(map< EcalLogicID, RunDCSHVDat >* fillMap )
  throw(std::runtime_error)
{
  this->checkConnection();

  fillMap->clear();

  try {
    std::pair< EcalLogicID, RunDCSHVDat > p;
    RunDCSHVDat dat;

    ResultSet* rset = getBarrelRset();
    
    fillTheMap(rset, fillMap);
    rset = getEndcapAnodeRset();
    
    fillTheMap(rset, fillMap);
    rset = getEndcapDynodeRset();
    
    fillTheMap(rset, fillMap);
  } 
  catch (SQLException &e) {
    throw(std::runtime_error("RunDCSHVDat::fetchData():  "+e.getMessage()));
  }
}

void RunDCSHVDat::fetchHistoricalData(std::list< std::pair<Tm, std::map< EcalLogicID, RunDCSHVDat > > >* fillMap, Tm timeStart  )
  throw(std::runtime_error)
{
  this->checkConnection();

  fillMap->clear();

  std::list<  DataReducer<RunDCSHVDat>::MyData<RunDCSHVDat>  > my_data_list;


  try {

    std::pair< EcalLogicID, RunDCSHVDat > p;
    RunDCSHVDat dat;

    ResultSet* rset1 = getBarrelRset(timeStart);
    fillTheMapByTime(rset1, &my_data_list);

    ResultSet* rset2 = getEndcapAnodeRset(timeStart);
    fillTheMapByTime(rset2, &my_data_list);

    ResultSet* rset3 = getEndcapDynodeRset(timeStart);
    fillTheMapByTime(rset3, &my_data_list);  


    DataReducer<RunDCSHVDat> my_dr;
    my_dr.setDataList(my_data_list);
    my_dr.getReducedDataList(fillMap);


  } 
  catch (SQLException &e) {
    throw(std::runtime_error("RunDCSHVDat::fetchData():  "+e.getMessage()));
  }
}
