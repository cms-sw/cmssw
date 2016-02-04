
//
// Gena Kukartsev (Brown), Feb 1, 2008
// $Id:
//
#include <limits>
#include <string>
#include <iostream>
#include <sstream>

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/DBInterface.h"

DBInterface::DBInterface (const std::string& fDb, bool fVerbose)
  : mConnect (0),
    mVerbose (fVerbose)
{
  mEnvironment = oracle::occi::Environment::createEnvironment (oracle::occi::Environment::OBJECT);
  // decode connect string
  size_t ipass = fDb.find ('/');
  size_t ihost = fDb.find ('@');

  if (ipass == std::string::npos || ihost == std::string::npos) {
    std::cerr << "DBInterface::DBInterface-> Error in connection std::string format: " << fDb
              << " Expect user/password@db" << std::endl;
  }
  else {
    std::string user (fDb, 0, ipass);
    std::string pass (fDb, ipass+1, ihost-ipass-1);
    std::string host (fDb, ihost+1);
    //    std::cout << "DBInterface::DBInterface-> Connecting " << user << '/' << pass << '@' << host << std::endl;
    try {
      mConnect = mEnvironment->createConnection(user, pass, host);
      mStatement = mConnect->createStatement ();
    }
    catch (oracle::occi::SQLException& sqlExcp) {
      std::cerr << "DBInterface::DBInterface exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
    }
  }
}

DBInterface::~DBInterface () {
  delete mStatement;
  mEnvironment->terminateConnection (mConnect);
  oracle::occi::Environment::terminateEnvironment (mEnvironment);
}

