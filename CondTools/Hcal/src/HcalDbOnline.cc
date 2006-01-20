
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbOnline.cc,v 1.1 2006/01/19 01:32:02 fedor Exp $
//
#include <string>
#include <iostream>

#include "occi.h" 

#include "CondTools/Hcal/interface/HcalDbOnline.h"


HcalDbOnline::HcalDbOnline (const std::string& fDb) 
  : mConnect (0)
{
  mEnvironment = oracle::occi::Environment::createEnvironment (oracle::occi::Environment::OBJECT);
  // decode connect string
  unsigned ipass = fDb.find ('/');
  unsigned ihost = fDb.find ('@');
  
  if (ipass == std::string::npos || ihost == std::string::npos) {
    std::cerr << "HcalDbOnline::HcalDbOnline-> Error in connection string format: " << fDb
	      << " Expect user/password@db" << std::endl;
  }
  else {
    std::string user (fDb, 0, ipass);
    std::string pass (fDb, ipass+1, ihost-ipass-1);
    std::string host (fDb, ihost+1);
    std::cout << "HcalDbOnline::HcalDbOnline-> Connecting " << user << '/' << pass << '@' << host << std::endl;
    try {
      mConnect = mEnvironment->createConnection(user, pass, host);
    }
    catch (oracle::occi::SQLException& sqlExcp) {
      std::cerr << "HcalDbOnline::HcalDbOnline exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
    }
  }
}

HcalDbOnline::~HcalDbOnline () {
  mEnvironment->terminateConnection (mConnect);
  oracle::occi::Environment::terminateEnvironment (mEnvironment);
}

std::auto_ptr <HcalPedestals> HcalDbOnline::getPedestals (const std::string& fTag) {
  std::auto_ptr <HcalPedestals> result (new HcalPedestals ());

  std::string query = "select ";
  query += "DAT.CAPACITOR_0_VALUE, DAT.CAPACITOR_1_VALUE, DAT.CAPACITOR_2_VALUE, DAT.CAPACITOR_3_VALUE, ";
  query += "CH.Z, CH.ETA, CH.PHI, CH.DEPTH, CH.DETECTOR_NAME ";
  query += "from ";
  query += "CMS_HCL_HCAL_CONDITION_OWNER.HCAL_GAIN_PEDSTL_CALIBRATIONS DAT, ";
  query += "CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS, ";
  query += "CMS_HCL_HCAL_CONDITION_OWNER.HCAL_CHANNELS CH, ";
  query += "CMS_HCL_CORE_CONDITION_OWNER.KINDS_OF_CONDITIONS KOC, ";
  query += "CMS_HCL_CORE_CONDITION_OWNER.COND_RUNS RN ";
  query += "where ";
  query += "DS.CONDITION_DATA_SET_ID=DAT.CONDITION_DATA_SET_ID and ";
  query += "DS.CHANNEL_MAP_ID = CH.CHANNEL_MAP_ID and ";
  query += "KOC.KIND_OF_CONDITION_ID = DS.KIND_OF_CONDITION_ID and ";
  query += "RN.COND_RUN_ID=DS.COND_RUN_ID and ";
  query += "KOC.IS_RECORD_DELETED='F' and ";
  query += "DS.IS_RECORD_DELETED='F' and ";
  query += "KOC.NAME='HCAL Pedestals' and ";
  query += "RN.RUN_NAME='" + fTag + "'";
  
  try {
    std::cout << "executing query: \n" << query << std::endl;
    oracle::occi::Statement* stmt = mConnect->createStatement ();
    stmt->setPrefetchRowCount (10000);
    stmt->setSQL (query);
    oracle::occi::ResultSet* rset = stmt->executeQuery ();
    while (rset->next ()) {
      float value [4];
      value [0] = rset->getFloat (1);
      value [1] = rset->getFloat (2);
      value [2] = rset->getFloat (3);
      value [3] = rset->getFloat (4);
      int z = rset->getInt (5);
      int eta = rset->getInt (6);
      int phi = rset->getInt (7);
      int depth = rset->getInt (8);
      std::string subdet = rset->getString (9);

      std::cout << "getting data: " <<  value [0] << '/' <<  value [1] << '/' <<  value [2] << '/' <<  value [3]
 		<< '/' <<  z << '/' <<  eta << '/' <<  phi << '/' <<  depth << '/' <<  subdet << std::endl;
      HcalSubdetector sub = subdet == "HB" ? HcalBarrel : 
	subdet == "HE" ? HcalEndcap :
	subdet == "HO" ? HcalOuter :
	subdet == "HF" ? HcalForward :  HcalSubdetector (0);
      HcalDetId id (sub, z * eta, phi, depth);
      result->addValue (id, value);
    }
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbOnline::getPedestals exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  result->sort ();
  return result;
}  

std::auto_ptr <HcalGains> HcalDbOnline::getGains (const std::string& fTag) {
  std::auto_ptr <HcalGains> result (new HcalGains ());
  return result;
}
