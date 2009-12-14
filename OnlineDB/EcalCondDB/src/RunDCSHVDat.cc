#include <stdexcept>
#include <string>
#include <math.h>

#include "OnlineDB/EcalCondDB/interface/RunDCSHVDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

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
}



RunDCSHVDat::~RunDCSHVDat()
{
}



void RunDCSHVDat::prepareWrite()
  throw(runtime_error)
{


}



void RunDCSHVDat::writeDB(const EcalLogicID* ecid, const RunDCSHVDat* item, RunIOV* iov)
  throw(runtime_error)
{
}



void RunDCSHVDat::fetchData(map< EcalLogicID, RunDCSHVDat >* fillMap, RunIOV* iov)
  throw(runtime_error)
{
  fetchLastData(fillMap);

}

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
    throw(runtime_error("RunDCSHVDat::getBarrelRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}

ResultSet *RunDCSHVDat::getEndcapAnodeRset() {
  ResultSet* rset = NULL;
  string query = "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
    " d.value, '600' NOMINAL_VALUE , d.since "
    "FROM "+ getEEAccount()+".DCSLASTVALUE_HV_VMON d "
    " JOIN "+ getEEAccount()+".EE_HVA_MAPPING h on "
    " h.DPID = d.DPID join channelview cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name"; 
  try {
    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  }
  catch (SQLException e) {
    throw(runtime_error("RunDCSHVDat::getEndcapAnodeRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}

ResultSet *RunDCSHVDat::getEndcapDynodeRset() {
  ResultSet* rset = NULL;
  string query = "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
    " d.value, '800' NOMINAL_VALUE , d.since "
    "FROM "+ getEEAccount()+".DCSLASTVALUE_HV_VMON d "
    " JOIN "+ getEEAccount()+".EE_HVD_MAPPING h on "
    " h.DPID = d.DPID join channelview cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name";
  try {
    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  } 
  catch (SQLException e) {
    throw(runtime_error("RunDCSHVDat::getEndcapDynodeRset():  " + e.getMessage() + " " + query));
  }
  return rset;
}

void RunDCSHVDat::fillTheMap(ResultSet *rset, 
			       map< EcalLogicID, RunDCSHVDat >* fillMap) {
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
    throw(runtime_error("RunDCSHVDat::fetchData():  "+e.getMessage()));
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
  if (int d = ((int)t_now_gmt_micros - (int)sinceTm.microsTime()) > 
      maxDifference) {
    dat.setStatus( -d/1000000 ); 
  }
  if (fabs(dat.getHV() - dat.getHVNominal())*1000 > maxHVDifferenceEB) {
    dat.setStatus(HVNOTNOMINAL);
  }
  if (dat.getHV()*1000 < minHV) {
    dat.setStatus(HVOFF);
  }
}

void  RunDCSHVDat::setStatusForEndcaps(RunDCSHVDat &dat, Tm sinceTm) {
  int t_now_gmt_micros = nowMicroseconds();
  if (int d = ((int)t_now_gmt_micros - (int)sinceTm.microsTime()) > 
      maxDifference) {
    dat.setStatus( -d/1000000 ); 
  }
  if (fabs(dat.getHV() - dat.getHVNominal())*1000 > maxHVDifferenceEE) {
    dat.setStatus(HVNOTNOMINAL);
  }
  if (dat.getHV()*1000 < minHV) {
    dat.setStatus(HVOFF);
  }
}

void RunDCSHVDat::fetchLastData(map< EcalLogicID, RunDCSHVDat >* fillMap )
  throw(runtime_error)
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
    throw(runtime_error("RunDCSHVDat::fetchData():  "+e.getMessage()));
  }
}
