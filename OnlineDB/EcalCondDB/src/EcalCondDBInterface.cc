// $Id: EcalCondDBInterface.cc,v 1.35 2011/09/14 13:27:59 organtin Exp $

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <cstdlib>
#include <time.h>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/IDBObject.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/DCSPTMTempList.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "OnlineDB/EcalCondDB/interface/MonRunList.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"

using namespace std;
using namespace oracle::occi;

void EcalCondDBInterface::fillLogicId2DetIdMaps() {
  // retrieve the lists of logic_ids, to build the detids
  std::vector<EcalLogicID> crystals_EB  =
    getEcalLogicIDSetOrdered( "EB_crystal_angle",
			      -85,85,1,360,
			      EcalLogicID::NULLID,EcalLogicID::NULLID,
			      "EB_crystal_number", 4 );
  std::vector<EcalLogicID> crystals_EE  =
    getEcalLogicIDSetOrdered( "EE_crystal_number",
			      -1,1,1,100,
			      1,100,
			      "EE_crystal_number", 4 );
  // fill the barrel map
  std::vector<EcalLogicID>::const_iterator ieb = crystals_EB.begin();
  std::vector<EcalLogicID>::const_iterator eeb = crystals_EB.end();
  while (ieb != eeb) {
    int iEta = ieb->getID1();
    int iPhi = ieb->getID2();
    EBDetId ebdetid(iEta,iPhi);
    _logicId2DetId[ieb->getLogicID()] = ebdetid;
    _detId2LogicId[ebdetid] = ieb->getLogicID();
    ieb++;
  }

  // fill the endcap map
  std::vector<EcalLogicID>::const_iterator iee = crystals_EE.begin();
  std::vector<EcalLogicID>::const_iterator eee = crystals_EE.end();

  while (iee != eee) {
    int iSide = iee->getID1();
    int iX    = iee->getID2();
    int iY    = iee->getID3();
    EEDetId eedetidpos(iX,iY,iSide);
    _logicId2DetId[iee->getLogicID()] = eedetidpos;
    _detId2LogicId[eedetidpos] = iee->getLogicID();
    iee++;
  }
  
}


EcalLogicID EcalCondDBInterface::getEcalLogicID( int logicID )
  throw(std::runtime_error)
{

  string sql = "SELECT name, logic_id, id1, id2, id3, maps_to FROM channelView WHERE logic_id = :logicID AND name=maps_to";
  
  int id1, id2, id3;
  string name, mapsTo;
  
  try {
    stmt->setSQL(sql);
    stmt->setInt(1, logicID);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      name = rset->getString(1);
      logicID = rset->getInt(2);
      id1 = rset->getInt(3);
      if (rset->isNull(3)) { id1 = EcalLogicID::NULLID; }
      id2 = rset->getInt(4);
      if (rset->isNull(4)) { id2 = EcalLogicID::NULLID; }
      id3 = rset->getInt(5);
      if (rset->isNull(5)) { id3 = EcalLogicID::NULLID; }
      mapsTo = rset->getString(6);
    } else {
      stringstream msg;
      msg << "ERROR:  Cannot build EcalLogicID for logic_id " << logicID;
      throw(std::runtime_error(msg.str()));
    }

  } catch (SQLException &e) {    
    throw(std::runtime_error("ERROR:  Failed to retrive ids:  " + e.getMessage() ));
  }
  
  return EcalLogicID( name, logicID, id1, id2, id3, mapsTo );  
}

std::list<ODDelaysDat> EcalCondDBInterface::fetchFEDelaysForRun(RunIOV *iov)
  throw(std::runtime_error)
{
  std::list<ODDelaysDat> ret;
  RunFEConfigDat d;
  std::map<EcalLogicID, RunFEConfigDat > fillMap;
  try {
    d.setConnection(env, conn);
    d.fetchData(&fillMap, iov);
  } catch (std::runtime_error &e) {
    throw e;
  }
  std::map<EcalLogicID, RunFEConfigDat >::const_iterator i = fillMap.begin();
  std::map<EcalLogicID, RunFEConfigDat >::const_iterator e = fillMap.end();
  while (i != e) {
    ODFEDAQConfig feDaqConfig;
    ODFEDAQConfig temp;
    temp.setId(i->second.getConfigId());
    feDaqConfig.setConnection(env, conn);
    feDaqConfig.fetchData(&temp);
    std::vector<ODDelaysDat> delays;
    ODDelaysDat temp2;
    temp2.setConnection(env, conn);
    temp2.fetchData(&delays, temp.getDelayId());
    std::vector<ODDelaysDat>::const_iterator di = delays.begin();
    std::vector<ODDelaysDat>::const_iterator de = delays.end();
    while (di != de) {
      ret.push_back(*di++);
    }
    i++;
  }
  return ret;
}

EcalLogicID EcalCondDBInterface::getEcalLogicID( string name,
						 int id1,
						 int id2,
						 int id3,
						 string mapsTo )
  throw(std::runtime_error)
{

  if (mapsTo == "") {
    mapsTo = name;
  }

  // build the sql string
  stringstream ss;
  ss << "SELECT logic_id FROM channelView WHERE name = :n AND";
  int idarray[] = {id1, id2, id3};
  for (int i=1; i<=3; i++) {
    if (idarray[i-1] == EcalLogicID::NULLID) {
      ss << " id"<<i<<" IS NULL AND";
    } else {
      ss << " id"<<i<<" = :id"<<i<<" AND";
    }
  }
  ss <<" maps_to = :m";
  
  // cout << "SQL:  " << ss.str() << endl;

  int logic_id;
  try {
    stmt->setSQL(ss.str());

    // bind the parameters
    int j = 1;  // parameter number counter
    stmt->setString(j, name);
    j++;
    for (int i=0; i<3; i++) {
      if (idarray[i] != EcalLogicID::NULLID) {
	stmt->setInt(j, idarray[i]);
	j++;
      }
    }
    stmt->setString(j, mapsTo);

    // execute the statement and retrieve the logic_id
    ResultSet* rset = stmt->executeQuery();
    if ( rset->next() ) {
      logic_id = rset->getInt(1);
    } else {
      stringstream msg;
      msg << "ERROR:  Query for EcalLogicID failed for parameters [" <<
	"name=" << name << ",maps_to=" << mapsTo << 
	",id1=" << id1 << ",id2=" << id2 << ",id3=" << id3 << "]";
      throw(std::runtime_error(msg.str()));
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("ERROR:  Failed to retrive logic_id:  " + e.getMessage() ));
  }

  // create and return the EcalLogicID object
  return EcalLogicID(name, logic_id, id1, id2, id3, mapsTo);
}

std::vector<EcalLogicID> EcalCondDBInterface::getEcalLogicIDSet( string name,
							    int fromId1, int toId1,
							    int fromId2, int toId2,
							    int fromId3, int toId3,
							    string mapsTo )
  throw(std::runtime_error)
{
  if (mapsTo == "") {
    mapsTo = name;
  }
  
  int idArray[] = { fromId1, toId1, fromId2, toId2, fromId3, toId3 };
  int from, to;
  
  stringstream ss;
  ss << "SELECT name, logic_id, id1, id2, id3, maps_to FROM channelView WHERE name = :name AND ";
  
  // loop through the three ids
  for (int i=1; i<=3; i++) {
    from = idArray[2*(i-1)];
    to   = idArray[2*(i-1) + 1];
    
    // check the id arguments in pairs
    if ((from == EcalLogicID::NULLID && to != EcalLogicID::NULLID) || // one is null
	(from != EcalLogicID::NULLID && to == EcalLogicID::NULLID) || //   but not the other
	(from > to)) { // negative interval
      throw(std::runtime_error("ERROR:  Bad arguments for getEcalLogicIDSet"));
    }
    
    // build the sql
    if (from == EcalLogicID::NULLID && to == EcalLogicID::NULLID) {
      ss << "id" << i << " IS NULL AND ";
    } else {
      ss << "id" << i << " >= :id" << i << "from AND " <<
	"id" << i << " <= :id" << i << "to AND ";
    }
  }
  ss << "maps_to = :maps_to ORDER BY id1, id2, id3";
  
  std::vector<EcalLogicID> result;
  
  try {
    stmt->setSQL(ss.str());

    // bind the parameters
    int j = 1;  // parameter number counter
    stmt->setString(j, name);
    j++;
    
    for (int i=0; i<3; i++) {
      from = idArray[2*i];
      to   = idArray[2*i + 1];
      if (from != EcalLogicID::NULLID) {
	stmt->setInt(j, from);
	j++;
	stmt->setInt(j, to);
	j++;
      }
    }

    stmt->setString(j, mapsTo);

  
    stmt->setPrefetchRowCount(IDBObject::ECALDB_NROWS);    

    ResultSet* rset = stmt->executeQuery();

    int id1, id2, id3, logicId;

    while (rset->next()) {
      name = rset->getString(1);
      logicId = rset->getInt(2);
      id1 = rset->getInt(3);
      if (rset->isNull(3)) { id1 = EcalLogicID::NULLID; }
      id2 = rset->getInt(4);
      if (rset->isNull(4)) { id2 = EcalLogicID::NULLID; }
      id3 = rset->getInt(5);
      if (rset->isNull(5)) { id3 = EcalLogicID::NULLID; }
      mapsTo = rset->getString(6);

      EcalLogicID ecid = EcalLogicID( name, logicId, id1, id2, id3, mapsTo );
      result.push_back(ecid);
    }
    stmt->setPrefetchRowCount(0);

  } catch (SQLException &e) {
    throw(std::runtime_error("ERROR:  Failure while getting EcalLogicID set:  " + e.getMessage() ));    
  }

  return result;
}

std::map<int, int> EcalCondDBInterface::getEcalLogicID2LmrMap() {
  std::map<int, int> ret;
  std::vector<EcalLogicID> crystals_EB  =
    getEcalLogicIDSetOrdered( "EB_crystal_number",
			      1,36,1,1700,
			      EcalLogicID::NULLID,EcalLogicID::NULLID,
			      "EB_crystal_number", EcalLogicID::NULLID);
  std::vector<EcalLogicID> crystals_EE  =
    getEcalLogicIDSetOrdered( "EE_crystal_number",
			      -1,1,1,100,
			      1,100,
			      "EE_crystal_number", EcalLogicID::NULLID);
  std::vector<EcalLogicID> EB_lmr  =
    getEcalLogicIDSetOrdered( "EB_crystal_number",
			      1,36,1,1700,
			      EcalLogicID::NULLID,EcalLogicID::NULLID,
			      "ECAL_LMR", EcalLogicID::NULLID);
  std::vector<EcalLogicID> EE_lmr  =
    getEcalLogicIDSetOrdered( "EE_crystal_number",
			      -1,1,1,100,
			      1,100,
			      "ECAL_LMR", EcalLogicID::NULLID);
  unsigned int neb = crystals_EB.size();
  unsigned int nee = crystals_EE.size();
  if (neb != EB_lmr.size()) {
    throw(std::runtime_error("ERROR: EB Vectors size do not agree"));
  }
  if (nee != EE_lmr.size()) {
    throw(std::runtime_error("ERROR: EE Vectors size do not agree"));
  }
  for (unsigned int i = 0; i < neb; i++) {
    ret[crystals_EB[i].getLogicID()] = EB_lmr[i].getLogicID() % 100;
  }
  for (unsigned int i = 0; i < nee; i++) {
    ret[crystals_EE[i].getLogicID()] = EE_lmr[i].getLogicID() % 100;
  }
  return ret;
}

std::vector<EcalLogicID> EcalCondDBInterface::getEcalLogicIDMappedTo(int lmr_logic_id, std::string maps_to) {
  std::string name = "EB_crystal_angle";
  std::string sql = "SELECT LOGIC_ID, ID1, ID2, ID3 "
    "FROM CHANNELVIEW WHERE NAME = 'EB_crystal_angle' AND LOGIC_ID IN "
    "(SELECT LOGIC_ID FROM CHANNELVIEW WHERE NAME = 'EB_crystal_number' AND "
    "ID1*10000+ID2 IN (SELECT DISTINCT ID1*10000+ID2 FROM CHANNELVIEW "
    "WHERE LOGIC_ID = :1 AND NAME = 'EB_crystal_number' AND MAPS_TO = :2) "
    "AND NAME = MAPS_TO)"; 
  if ((lmr_logic_id / 1000000000) == 2) {
    name = "EE_crystal_number";
    sql = "SELECT LOGIC_ID, ID1, ID2, ID3 "
      "FROM CHANNELVIEW WHERE NAME = 'EE_crystal_number' AND LOGIC_ID IN "
      "(SELECT LOGIC_ID FROM CHANNELVIEW WHERE NAME = 'EE_crystal_number' AND "
      "ID1*10000000+ID2*10000+ID3 IN (SELECT DISTINCT "
      "ID1*10000000+ID2*10000+ID3 FROM CHANNELVIEW "
      "WHERE LOGIC_ID = :1 AND NAME = 'EE_crystal_number' AND MAPS_TO = :2) "
      "AND NAME = MAPS_TO) AND NAME = MAPS_TO"; 
  }
  std::vector<EcalLogicID> ret;
  try {
    stmt->setSQL(sql.c_str());
    stmt->setInt(1, lmr_logic_id);
    stmt->setString(2, maps_to);
    stmt->setPrefetchRowCount(IDBObject::ECALDB_NROWS);    
    
    ResultSet* rset = stmt->executeQuery();
    
    while (rset->next()) {
      int logic_id = rset->getInt(1);
      int id1 = rset->getInt(2);
      if (rset->isNull(2)) { id1 = EcalLogicID::NULLID; }
      int id2 = rset->getInt(3);
      if (rset->isNull(3)) { id2 = EcalLogicID::NULLID; }
      int id3 = rset->getInt(4);
      if (rset->isNull(4)) { id3 = EcalLogicID::NULLID; }
      
      EcalLogicID ecid = EcalLogicID( name, logic_id, id1, id2, id3, maps_to );
      ret.push_back(ecid);
    }
    stmt->setPrefetchRowCount(0);
  }
  catch (SQLException &e) {
    throw(std::runtime_error("ERROR:  Failure while getting EcalLogicID set:  " + e.getMessage() ));
  }
  return ret;
}

std::vector<EcalLogicID> EcalCondDBInterface::getEcalLogicIDForLMR(int lmr) {
  return getEcalLogicIDMappedTo(lmr, "ECAL_LMR");
}

std::vector<EcalLogicID> EcalCondDBInterface::getEcalLogicIDForLMR(const EcalLogicID &lmr) {
  return getEcalLogicIDForLMR(lmr.getLogicID());
}

std::vector<EcalLogicID> EcalCondDBInterface::getEcalLogicIDForLMPN(int lmr) {
  if ((lmr / 1000000000) == 2) {
    return getEcalLogicIDMappedTo(lmr, "EE_LM_PN");
  } else {
    return getEcalLogicIDMappedTo(lmr, "EB_LM_PN");
  }
}

std::vector<EcalLogicID> EcalCondDBInterface::getEcalLogicIDForLMPN(const EcalLogicID &lmr) {
  return getEcalLogicIDForLMR(lmr.getLogicID());
}

std::vector<EcalLogicID> EcalCondDBInterface::getEcalLogicIDSetOrdered( string name,
							    int fromId1, int toId1,
							    int fromId2, int toId2,
							    int fromId3, int toId3,
							    string mapsTo, int orderedBy )
  // the orderedBy can be 1, 2, 3, 4
  // corresponding to id1 id2 id3 or logic_id 
  throw(std::runtime_error)
{
  if (mapsTo == "") {
    mapsTo = name;
  }
  
  int idArray[] = { fromId1, toId1, fromId2, toId2, fromId3, toId3 };
  int from, to;
  
  stringstream ss;
  ss << "SELECT name, logic_id, id1, id2, id3, maps_to FROM channelView WHERE name = :name AND ";
  
  // loop through the three ids
  for (int i=1; i<=3; i++) {
    from = idArray[2*(i-1)];
    to   = idArray[2*(i-1) + 1];
    
    // check the id arguments in pairs
    if ((from == EcalLogicID::NULLID && to != EcalLogicID::NULLID) || // one is null
	(from != EcalLogicID::NULLID && to == EcalLogicID::NULLID) || //   but not the other
	(from > to)) { // negative interval
      throw(std::runtime_error("ERROR:  Bad arguments for getEcalLogicIDSet"));
    }
    
    // build the sql
    if (from == EcalLogicID::NULLID && to == EcalLogicID::NULLID) {
      ss << "id" << i << " IS NULL AND ";
    } else {
      ss << "id" << i << " >= :id" << i << "from AND " <<
	"id" << i << " <= :id" << i << "to AND ";
    }
  }
  ss << "maps_to = :maps_to ";

  if(orderedBy==EcalLogicID::NULLID){
    ss<<"  ORDER BY id1, id2, id3";
  } else if(orderedBy==1 || orderedBy==12 || orderedBy==123){
    ss<<"  ORDER BY id1, id2, id3 ";
  } else if (orderedBy==213 || orderedBy==21 ){ 
    ss<<"  ORDER BY id2, id1, id3 ";
  } else if (orderedBy==231|| orderedBy==23){ 
    ss<<"  ORDER BY id2, id3, id1 ";
  } else if (orderedBy==321|| orderedBy==32){
    ss<<"  ORDER BY id3, id2, id1 ";
  } else if (orderedBy==312|| orderedBy==31){
    ss<<"  ORDER BY id3, id1, id2 ";
  } else if (orderedBy==132|| orderedBy==13){ 
    ss<<"  ORDER BY id1, id3, id2 ";
  } else if (orderedBy==1234 ){ 
    ss<<"  ORDER BY id1, id2, id3, logic_id ";
  } else if (orderedBy==4) {
    ss<<"  ORDER BY logic_id ";
  } else {
    ss<<"  ORDER BY id1, id2, id3";
  }
  
  std::vector<EcalLogicID> result;
  
  try {
    stmt->setSQL(ss.str());

    // bind the parameters
    int j = 1;  // parameter number counter
    stmt->setString(j, name);
    j++;
    
    for (int i=0; i<3; i++) {
      from = idArray[2*i];
      to   = idArray[2*i + 1];
      if (from != EcalLogicID::NULLID) {
	stmt->setInt(j, from);
	j++;
	stmt->setInt(j, to);
	j++;
      }
    }

    stmt->setString(j, mapsTo);

  
    stmt->setPrefetchRowCount(IDBObject::ECALDB_NROWS);    

    ResultSet* rset = stmt->executeQuery();

    int id1, id2, id3, logicId;

    while (rset->next()) {
      name = rset->getString(1);
      logicId = rset->getInt(2);
      id1 = rset->getInt(3);
      if (rset->isNull(3)) { id1 = EcalLogicID::NULLID; }
      id2 = rset->getInt(4);
      if (rset->isNull(4)) { id2 = EcalLogicID::NULLID; }
      id3 = rset->getInt(5);
      if (rset->isNull(5)) { id3 = EcalLogicID::NULLID; }
      mapsTo = rset->getString(6);

      EcalLogicID ecid = EcalLogicID( name, logicId, id1, id2, id3, mapsTo );
      result.push_back(ecid);
    }
    stmt->setPrefetchRowCount(0);

  } catch (SQLException &e) {
    throw(std::runtime_error("ERROR:  Failure while getting EcalLogicID set:  " + e.getMessage() ));    
  }

  return result;
}



void EcalCondDBInterface::insertRunIOV(RunIOV* iov)
  throw(std::runtime_error)
{
  try {
    iov->setConnection(env, conn);
    iov->writeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::insertLmfSeq(LMFSeqDat *iov) 
  throw(std::runtime_error) 
{
  try {
    iov->setConnection(env, conn);
    iov->writeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::insertLmfLmrSubIOV(LMFLmrSubIOV *iov) 
  throw(std::runtime_error) 
{
  try {
    iov->setConnection(env, conn);
    iov->writeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::insertLmfIOV(LMFIOV *iov) 
  throw(std::runtime_error) 
{
  try {
    iov->setConnection(env, conn);
    iov->writeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::insertLmfDat(LMFDat *dat) 
  throw(std::runtime_error) 
{
  try {
    dat->setConnection(env, conn);
    dat->writeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::insertLmfDat(std::list<LMFDat *> dat) 
  throw(std::runtime_error) 
{
  try {
    std::list<LMFDat *>::iterator i = dat.begin();
    std::list<LMFDat *>::iterator e = dat.end();
    while (i != e) {
      (*i)->setConnection(env, conn);
      (*i)->writeDB();
      i++;
    }
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::insertLmfRunIOV(LMFRunIOV *iov) 
  throw(std::runtime_error) 
{
  try {
    iov->setConnection(env, conn);
    iov->writeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::updateRunIOV(RunIOV* iov)
  throw(std::runtime_error)
{
  try {
    iov->setConnection(env, conn);
    iov->updateEndTimeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::updateRunIOVEndTime(RunIOV* iov)
  throw(std::runtime_error)
{
  try {
    iov->setConnection(env, conn);
    iov->updateEndTimeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::updateRunIOVStartTime(RunIOV* iov)
  throw(std::runtime_error)
{
  try {
    iov->setConnection(env, conn);
    iov->updateStartTimeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::updateRunConfig(ODRunConfigInfo* od)
  throw(std::runtime_error)
{
  try {
    od->setConnection(env, conn);
    od->updateDefaultCycle();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

RunIOV EcalCondDBInterface::fetchRunIOV(RunTag* tag, run_t run)
  throw(std::runtime_error)
{  
  RunIOV iov;
  iov.setConnection(env, conn);
  iov.setByRun(tag, run);
  return iov;
}



RunIOV EcalCondDBInterface::fetchRunIOV(std::string location, run_t run)
  throw(std::runtime_error)
{  
  RunIOV iov;
  iov.setConnection(env, conn);
  iov.setByRun(location, run);
  return iov;
}

RunIOV EcalCondDBInterface::fetchRunIOV(std::string location, const Tm &t) 
  throw(std::runtime_error)
{
  RunIOV iov;
  iov.setConnection(env, conn);
  iov.setByTime(location, t);
  return iov;
}

void EcalCondDBInterface::insertMonRunIOV(MonRunIOV* iov)
  throw(std::runtime_error)
{
  try {
    iov->setConnection(env, conn);
    iov->writeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}

void EcalCondDBInterface::insertDCUIOV(DCUIOV* iov)
  throw(std::runtime_error)
{
  try {
    iov->setConnection(env, conn);
    iov->writeDB();
  } catch(std::runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}




MonRunIOV EcalCondDBInterface::fetchMonRunIOV(RunTag* runtag, MonRunTag* montag, run_t run, subrun_t subrun)
  throw(std::runtime_error)
{
  RunIOV runiov = fetchRunIOV(runtag, run);
  MonRunIOV moniov;
  moniov.setConnection(env, conn);
  moniov.setByRun(montag, &runiov, subrun);
  return moniov;
}



DCUIOV EcalCondDBInterface::fetchDCUIOV(DCUTag* tag, Tm eventTm)
  throw(std::runtime_error)
{
  DCUIOV dcuiov;
  dcuiov.setConnection(env, conn);
  dcuiov.setByTm(tag, eventTm);
  return dcuiov;
}

RunIOV EcalCondDBInterface::fetchLMFLastRun() const {
  LMFSeqDat seq(env, conn);
  return seq.fetchLastRun();
}

LMFRunIOV EcalCondDBInterface::fetchLMFRunIOV(RunTag* runtag, LMFRunTag* lmftag, run_t run, subrun_t subrun)
  throw(std::runtime_error)
{
  RunIOV runiov = fetchRunIOV(runtag, run);
  LMFRunIOV lmfiov;
  lmfiov.setConnection(env, conn);
  //  lmfiov.setByRun(lmftag, &runiov, subrun);
  return lmfiov;
}

bool EcalCondDBInterface::fetchLMFRunIOV(const LMFSeqDat &seq, LMFRunIOV& iov, 
					 int lmr, int type, int color ) const {
  bool ret = false;
  iov.setConnection(env, conn);
  std::list<LMFRunIOV> iovlist = iov.fetchBySequence(seq, lmr, type, color);
  int s = iovlist.size();
  if (s > 0) {
    iov = iovlist.front();
    ret = true;
    if (s > 1) {
      // should not happen
      std::cout << "################################" << std::endl;
      std::cout << "################################" << std::endl;
      std::cout << "WARNING: More than one LMFRUNIOV" << std::endl;
      std::cout << "         Found for seq " << seq.getID() << std::endl;
      std::cout << "         lmr " << lmr << " type " << type << std::endl;
      std::cout << "         and color " << color << std::endl;
      std::cout << "################################" << std::endl;
      std::cout << "################################" << std::endl;
    }
  } else {
    // find the most recent data
    iovlist = iov.fetchLastBeforeSequence(seq, lmr, type, color);
    s = iovlist.size();
    if (s == 1) {
      iov = iovlist.front();
    } 
  }
  return ret;
}

CaliIOV EcalCondDBInterface::fetchCaliIOV(CaliTag* tag, Tm eventTm)
  throw(std::runtime_error)
{
  CaliIOV caliiov;
  caliiov.setConnection(env, conn);
  caliiov.setByTm(tag, eventTm);
  return caliiov;
}

DCSPTMTempList EcalCondDBInterface::fetchDCSPTMTempList(EcalLogicID ecid)
  throw(std::runtime_error)
{  
  DCSPTMTempList r;
  r.setConnection(env, conn);
  r.fetchValuesForECID(ecid);
  return r;
}

DCSPTMTempList EcalCondDBInterface::fetchDCSPTMTempList(EcalLogicID ecid, Tm start, Tm end)
  throw(std::runtime_error)
{  
  DCSPTMTempList r;
  r.setConnection(env, conn);
  r.fetchValuesForECIDAndTime(ecid, start, end);
  return r;
}

RunList EcalCondDBInterface::fetchRunList(RunTag tag)
  throw(std::runtime_error)
{  
  RunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.fetchRuns();
  return r;
}

RunList EcalCondDBInterface::fetchRunList(RunTag tag, int min_run, int max_run) throw(std::runtime_error){
  RunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.fetchRuns( min_run,  max_run);
  return r;
}

RunList EcalCondDBInterface::fetchNonEmptyRunList(RunTag tag, int min_run, int max_run) throw(std::runtime_error){
  RunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.fetchNonEmptyRuns( min_run,  max_run);
  return r;
}

RunList EcalCondDBInterface::fetchNonEmptyGlobalRunList(RunTag tag, int min_run, int max_run) throw(std::runtime_error){
  RunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.fetchNonEmptyGlobalRuns( min_run,  max_run);
  return r;
}

RunList EcalCondDBInterface::fetchRunListByLocation(RunTag tag, int min_run, int max_run , const LocationDef locDef) 
  throw(std::runtime_error) {
  RunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.fetchRunsByLocation( min_run,  max_run, locDef);
  return r;
}

RunList EcalCondDBInterface::fetchGlobalRunListByLocation(RunTag tag, int min_run, int max_run , const LocationDef locDef) 
  throw(std::runtime_error) {
  RunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.fetchGlobalRunsByLocation( min_run,  max_run, locDef);
  return r;
}

RunList EcalCondDBInterface::fetchRunListLastNRuns(RunTag tag, int max_run, int n_runs) 
  throw(std::runtime_error){
  RunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.fetchLastNRuns( max_run,  n_runs);
  return r;
}




// from here it is for the MonRunList 

MonRunList EcalCondDBInterface::fetchMonRunList(RunTag tag, MonRunTag monrunTag)
  throw(std::runtime_error)
{  
  MonRunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.setMonRunTag(monrunTag);
  r.fetchRuns();
  return r;
}

MonRunList EcalCondDBInterface::fetchMonRunList(RunTag tag, MonRunTag monrunTag,int min_run, int max_run)
  throw(std::runtime_error)
{  
  MonRunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.setMonRunTag(monrunTag);
  r.fetchRuns(min_run, max_run);
  return r;
}

MonRunList EcalCondDBInterface::fetchMonRunListLastNRuns(RunTag tag, MonRunTag monrunTag,int max_run, int n_runs )
  throw(std::runtime_error)
{  
  MonRunList r;
  r.setConnection(env, conn);
  r.setRunTag(tag);
  r.setMonRunTag(monrunTag);
  r.fetchLastNRuns(max_run, n_runs );
  return r;
}



void EcalCondDBInterface::dummy()
{
}
