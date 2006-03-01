// $Id: EcalCondDBInterface.cc,v 1.7 2006/02/10 21:59:43 egeland Exp $

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <cstdlib>
#include <time.h>
#include <stdexcept>
#include <occi.h>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;


EcalLogicID EcalCondDBInterface::getEcalLogicID( int logicID )
  throw(runtime_error)
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
      throw(runtime_error(msg.str()));
    }

  } catch (SQLException &e) {    
    throw(runtime_error("ERROR:  Failed to retrive ids:  " + e.getMessage() ));
  }
  
  return EcalLogicID( name, logicID, id1, id2, id3, mapsTo );  
}



EcalLogicID EcalCondDBInterface::getEcalLogicID( string name,
						 int id1,
						 int id2,
						 int id3,
						 string mapsTo )
  throw(runtime_error)
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
      throw(runtime_error(msg.str()));
    }
  } catch (SQLException &e) {
    throw(runtime_error("ERROR:  Failed to retrive logic_id:  " + e.getMessage() ));
  }

  // create and return the EcalLogicID object
  return EcalLogicID(name, logic_id, id1, id2, id3, mapsTo);
}



vector<EcalLogicID> EcalCondDBInterface::getEcalLogicIDSet( string name,
							    int fromId1, int toId1,
							    int fromId2, int toId2,
							    int fromId3, int toId3,
							    string mapsTo )
  throw(runtime_error)
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
      throw(runtime_error("ERROR:  Bad arguments for getEcalLogicIDSet"));
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
  
  vector<EcalLogicID> result;
  
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

  } catch (SQLException &e) {
    throw(runtime_error("ERROR:  Failure while getting EcalLogicID set:  " + e.getMessage() ));    
  }

  return result;
}



void EcalCondDBInterface::insertRunIOV(RunIOV* iov)
  throw(runtime_error)
{
  try {
    iov->setConnection(env, conn);
    iov->writeDB();
  } catch(runtime_error &e) {
    conn->rollback();
    throw(e);
  }
  conn->commit();
}



RunIOV EcalCondDBInterface::fetchRunIOV(RunTag* tag, run_t run)
  throw(runtime_error)
{  
  RunIOV iov;
  iov.setConnection(env, conn);
  iov.setByRun(tag, run);
  return iov;
}



MonRunIOV EcalCondDBInterface::fetchMonRunIOV(RunTag* runtag, MonRunTag* montag, run_t run, subrun_t subrun)
  throw(runtime_error)
{
  RunIOV runiov = fetchRunIOV(runtag, run);
  MonRunIOV moniov;
  moniov.setConnection(env, conn);
  moniov.setByRun(montag, &runiov, subrun);
  return moniov;
}



DCUIOV EcalCondDBInterface::fetchDCUIOV(DCUTag* tag, Tm eventTm)
  throw(runtime_error)
{
  DCUIOV dcuiov;
  dcuiov.setConnection(env, conn);
  dcuiov.setByTm(tag, eventTm);
  return dcuiov;
}



LMFRunIOV EcalCondDBInterface::fetchLMFRunIOV(RunTag* runtag, LMFRunTag* lmftag, run_t run, subrun_t subrun)
  throw(runtime_error)
{
  RunIOV runiov = fetchRunIOV(runtag, run);
  LMFRunIOV lmfiov;
  lmfiov.setConnection(env, conn);
  lmfiov.setByRun(lmftag, &runiov, subrun);
  return lmfiov;
}



void EcalCondDBInterface::dummy()
{
}
