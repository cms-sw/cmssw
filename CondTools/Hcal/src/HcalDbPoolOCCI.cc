
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbPoolOCCI.cc,v 1.7 2010/11/29 20:41:57 wmtan Exp $
//
#include <string>
#include <iostream>
#include <cstdio>

#include "OnlineDB/Oracle/interface/Oracle.h" 

#include "CondTools/Hcal/interface/HcalDbPoolOCCI.h"

const bool debug = false;

namespace {
  long getObjectId (const std::string& fToken) {
    size_t ipos = fToken.find ("OID=");
    if (ipos != std::string::npos) {
      ipos = fToken.find ('-', ipos);
      if (ipos != std::string::npos) {
        size_t ipos2 = fToken.find (']', ipos);
	if (ipos2 != std::string::npos) {
	  while (fToken [++ipos] != '0');
	  std::string id (fToken, ipos, ipos2-ipos);
	  char* endptr = 0;
	  unsigned long result = strtoul (id.c_str (), &endptr, 16);
	  if (endptr && !*endptr) return long (result);
	}
      }
    }
    return -1;
  }
  
  const char* getTable (const HcalPedestals* fObject) {return "HCALPEDESTAL";}
  const char* getTable (const HcalGains* fObject) {return "HCALGAIN";}
}

HcalDbPoolOCCI::HcalDbPoolOCCI (const std::string& fDb) 
  : mConnect (0)
{
  mEnvironment = oracle::occi::Environment::createEnvironment (oracle::occi::Environment::OBJECT);
  // decode connect string
  size_t ipass = fDb.find ('/');
  size_t ihost = fDb.find ('@');
  
  if (ipass == std::string::npos || ihost == std::string::npos) {
    std::cerr << "HcalDbPoolOCCI::HcalDbPoolOCCI-> Error in connection string format: " << fDb
	      << " Expect user/password@db" << std::endl;
  }
  else {
    std::string user (fDb, 0, ipass);
    std::string pass (fDb, ipass+1, ihost-ipass-1);
    std::string host (fDb, ihost+1);
    //     if (debug) std::cout << "HcalDbPoolOCCI::HcalDbPoolOCCI-> Connecting " << user << '/' << pass << '@' << host << std::endl;
    try {
      mConnect = mEnvironment->createConnection(user, pass, host);
      mStatement = mConnect->createStatement ();
    }
    catch (oracle::occi::SQLException& sqlExcp) {
      std::cerr << "HcalDbPoolOCCI::HcalDbPoolOCCI exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
    }
  }
}

HcalDbPoolOCCI::~HcalDbPoolOCCI () {
  delete mStatement;
  mEnvironment->terminateConnection (mConnect);
  oracle::occi::Environment::terminateEnvironment (mEnvironment);
}

bool HcalDbPoolOCCI::getObject (HcalPedestals* fObject, const std::string& fTag, unsigned long fRun) {
  HcalPedestal* myped(0);
  return getObjectGeneric (fObject, myped, fTag, fRun);
}

bool HcalDbPoolOCCI::getObject (HcalGains* fObject, const std::string& fTag, unsigned long fRun) {
  HcalGain* mygain(0);
  return getObjectGeneric (fObject, mygain, fTag, fRun);
}

bool HcalDbPoolOCCI::getObject (HcalElectronicsMap* fObject, const std::string& fTag, unsigned long fRun) {
  return false;
}



std::string HcalDbPoolOCCI::getMetadataToken (const std::string& fTag) {
  std::string result = "";
  std::string sql_query = "select * from metadata";
  try {
     if (debug) std::cout << "executing query: \n" << sql_query << std::endl;
    mStatement->setPrefetchRowCount (100);
    mStatement->setSQL (sql_query);
    oracle::occi::ResultSet* rset = mStatement->executeQuery ();
    while (rset->next ()) {
      std::string name = rset->getString (1);
      std::string token = rset->getString (2);
      if (name == fTag) {
	result = token;
	break;
      }
    }
    delete rset;
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbPoolOCCI::getMetadataToken exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  return result;
}

std::string HcalDbPoolOCCI::getDataToken (const std::string& fIov, unsigned long fRun) {
  std::string result = "";
  long iovId = getObjectId (fIov);
  if (iovId >= 0) {
    char sql_query [1024];
    sprintf (sql_query, "select IOV_IOV_UNSIGNED_LONG, IOV_IOV_STRING from COND__IOV_IOV where ID_ID = %ld ORDER BY IOV_IOV_UNSIGNED_LONG DESC", iovId);
    try {
       if (debug) std::cout << "executing query: \n" << sql_query << std::endl;
      mStatement->setPrefetchRowCount (100);
      mStatement->setSQL (std::string (sql_query));
      oracle::occi::ResultSet* rset = mStatement->executeQuery ();
      while (rset->next ()) {
	unsigned long runMax = rset->getUInt (1);
	std::string token = rset->getString (2);
	if (fRun <= runMax) {
	  result = token;
	  break;
	}
      }
      delete rset;
    }
    catch (oracle::occi::SQLException& sqlExcp) {
      std::cerr << "HcalDbPoolOCCI::getDataToken exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
    }
  }
  return result;
}

template <class T, class S>
bool HcalDbPoolOCCI::getObjectGeneric (T* fObject, S* fCondObject, const std::string& fTag, unsigned long fRun) {
  std::string mdToken = getMetadataToken (fTag);
   if (debug) std::cout << "HcalDbPoolOCCI::getObjectGeneric-> tag/token: " << fTag << '/' << mdToken << std::endl;
  if (mdToken.empty ()) return false;
  std::string objToken = getDataToken (mdToken, fRun);
   if (debug) std::cout << "HcalDbPoolOCCI::getObjectGeneric-> Run/token: " << fRun << '/' << objToken << std::endl;
  if (objToken.empty ()) return false;
  long id = getObjectId (objToken);
  if (id >= 0) {
    char sql_query [1024];
    const char* name = getTable (fObject);
    sprintf (sql_query, "select MITEMS_%s_MID, MITEMS_%s_MVALUE1, MITEMS_%s_MVALUE2, MITEMS_%s_MVALUE3, MITEMS_%s_MVALUE4 from %sS_MITEMS where ID_ID = %ld ORDER BY MITEMS_%s_MID",
 	     name, name, name, name, name, name, id, name);
    try {
       if (debug) std::cout << "executing query: \n" << sql_query << std::endl;
      mStatement->setPrefetchRowCount (100);
      mStatement->setSQL (sql_query);
      oracle::occi::ResultSet* rset = mStatement->executeQuery ();
      while (rset->next ()) {
	unsigned long hcalId = rset->getUInt (1);
	float values [4];
	for (int i = 0; i < 4; i++) values [i] = rset->getFloat (i+2);

	fCondObject = new S(DetId (hcalId), values[0], values[1], values[2], values[3]);
	fObject->addValues (*fCondObject);
	delete fCondObject;
	//	 if (debug) std::cout << "new entry: " << hcalId << '/' << values [0] << '/' << values [1] << '/' 
	//	  << values [2] << '/' << values [3] << std::endl;
      }
      delete rset;
      return true;
    }
    catch (oracle::occi::SQLException& sqlExcp) {
      std::cerr << "HcalDbPoolOCCI::getObjectn exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
    }
  }
  return false;
}
