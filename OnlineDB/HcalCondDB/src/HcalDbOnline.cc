
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbOnline.cc,v 1.21 2010/11/29 20:41:57 wmtan Exp $
//
#include <limits>
#include <string>
#include <iostream>
#include <sstream>

#include "OnlineDB/Oracle/interface/Oracle.h" 

#include "FWCore/Utilities/interface/Exception.h"
#include "OnlineDB/HcalCondDB/interface/HcalDbOnline.h"

namespace {

  HcalSubdetector hcalSubdet (const std::string& fName) {
    return fName == "HB" ? HcalBarrel : 
      fName == "HE" ? HcalEndcap :
      fName == "HO" ? HcalOuter :
      fName == "HF" ? HcalForward :  HcalSubdetector (0);
  }
}

HcalDbOnline::HcalDbOnline (const std::string& fDb, bool fVerbose) 
  : mConnect (0),
    mVerbose (fVerbose)
{
  mEnvironment = oracle::occi::Environment::createEnvironment (oracle::occi::Environment::OBJECT);
  // decode connect string
  size_t ipass = fDb.find ('/');
  size_t ihost = fDb.find ('@');
  
  if (ipass == std::string::npos || ihost == std::string::npos) {
    std::cerr << "HcalDbOnline::HcalDbOnline-> Error in connection string format: " << fDb
	      << " Expect user/password@db" << std::endl;
  }
  else {
    std::string user (fDb, 0, ipass);
    std::string pass (fDb, ipass+1, ihost-ipass-1);
    std::string host (fDb, ihost+1);
    //    std::cout << "HcalDbOnline::HcalDbOnline-> Connecting " << user << '/' << pass << '@' << host << std::endl;
    try {
      mConnect = mEnvironment->createConnection(user, pass, host);
      mStatement = mConnect->createStatement ();
    }
    catch (oracle::occi::SQLException& sqlExcp) {
      std::cerr << "HcalDbOnline::HcalDbOnline exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
    }
  }
}

HcalDbOnline::~HcalDbOnline () {
  delete mStatement;
  mEnvironment->terminateConnection (mConnect);
  oracle::occi::Environment::terminateEnvironment (mEnvironment);
}

std::vector<std::string> HcalDbOnline::metadataAllTags () {
  std::vector<std::string> result;
  std::string sql_query ("");
  sql_query += "SELECT unique TAG_NAME from V_TAG_IOV_CONDDATASET order by TAG_NAME\n"; 
  try {
    if (mVerbose) std::cout << "executing query: \n" << sql_query << std::endl;
    mStatement->setPrefetchRowCount (100);
    mStatement->setSQL (sql_query);
    oracle::occi::ResultSet* rset = mStatement->executeQuery ();
    while (rset->next ()) {
      std::string tag = rset->getString (1);
      result.push_back (tag);
    }
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbOnline::metadataAllTags exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  return result;
}

std::vector<HcalDbOnline::IntervalOV> HcalDbOnline::getIOVs (const std::string& fTag) {
  std::vector<IntervalOV> result;
  std::string sql_query ("");
  sql_query += "SELECT unique INTERVAL_OF_VALIDITY_BEGIN, INTERVAL_OF_VALIDITY_END from V_TAG_IOV_CONDDATASET\n";
  sql_query += "WHERE TAG_NAME='" + fTag + "'\n";
  sql_query += "ORDER by INTERVAL_OF_VALIDITY_BEGIN\n";
  try {
    if (mVerbose) std::cout << "executing query: \n" << sql_query << std::endl;
    mStatement->setPrefetchRowCount (100);
    mStatement->setSQL (sql_query);
    oracle::occi::ResultSet* rset = mStatement->executeQuery ();
    while (rset->next ()) {
//       char buffer [128];
//       oracle::occi::Bytes iovb = rset->getNumber (1).toBytes();
//       unsigned ix = 0;
//       std::cout << "total bytes: " << iovb.length() << std::endl;
//       for (; ix < iovb.length(); ix++) {
// 	sprintf (buffer, "byte# %d: %x", ix, iovb.byteAt (ix));
// 	std::cout << buffer << std::endl; 
//       }
      IOVTime beginIov = (unsigned long) rset->getNumber (1);
//       sprintf (buffer, "%x", beginIov);
//       std::cout << "value: " << buffer << std::endl;
      IOVTime endIov = rset->getInt (2);
      if (!endIov) endIov = std::numeric_limits <IOVTime>::max (); // end of ages
      result.push_back (std::make_pair (beginIov, endIov));
    }
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbOnline::getIOVs exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  return result;
}

