
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbOnline.cc,v 1.3 2006/01/21 01:35:39 fedor Exp $
//
#include <string>
#include <iostream>

#include "occi.h" 

#include "CondTools/Hcal/interface/HcalDbOnline.h"

namespace {
  std::string query (const HcalPedestals* fObject, const std::string& fTag) {
    std::string result = "select ";
    result += "DAT.CAPACITOR_0_VALUE, DAT.CAPACITOR_1_VALUE, DAT.CAPACITOR_2_VALUE, DAT.CAPACITOR_3_VALUE, ";
    result += "CH.Z, CH.ETA, CH.PHI, CH.DEPTH, CH.DETECTOR_NAME ";
    result += "from ";
    result += "CMS_HCL_HCAL_CONDITION_OWNER.HCAL_GAIN_PEDSTL_CALIBRATIONS DAT, ";
    result += "CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS, ";
    result += "CMS_HCL_HCAL_CONDITION_OWNER.HCAL_CHANNELS CH, ";
    result += "CMS_HCL_CORE_CONDITION_OWNER.KINDS_OF_CONDITIONS KOC, ";
    result += "CMS_HCL_CORE_CONDITION_OWNER.COND_RUNS RN ";
    result += "where ";
    result += "DS.CONDITION_DATA_SET_ID=DAT.CONDITION_DATA_SET_ID and ";
    result += "DS.CHANNEL_MAP_ID = CH.CHANNEL_MAP_ID and ";
    result += "KOC.KIND_OF_CONDITION_ID = DS.KIND_OF_CONDITION_ID and ";
    result += "RN.COND_RUN_ID=DS.COND_RUN_ID and ";
    result += "KOC.IS_RECORD_DELETED='F' and ";
    result += "DS.IS_RECORD_DELETED='F' and ";
    result += "KOC.NAME='HCAL Pedestals' and ";
    result += "RN.RUN_NAME='" + fTag + "'";
    return result;
  }
  std::string query (const HcalGains* fObject, const std::string& fTag) {
    std::string result = "select ";
    result += "DAT.CAPACITOR_0_VALUE, DAT.CAPACITOR_1_VALUE, DAT.CAPACITOR_2_VALUE, DAT.CAPACITOR_3_VALUE, ";
    result += "CH.Z, CH.ETA, CH.PHI, CH.DEPTH, CH.DETECTOR_NAME ";
    result += "from ";
    result += "CMS_HCL_HCAL_CONDITION_OWNER.HCAL_GAIN_PEDSTL_CALIBRATIONS DAT, ";
    result += "CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS, ";
    result += "CMS_HCL_HCAL_CONDITION_OWNER.HCAL_CHANNELS CH, ";
    result += "CMS_HCL_CORE_CONDITION_OWNER.KINDS_OF_CONDITIONS KOC, ";
    result += "CMS_HCL_CORE_CONDITION_OWNER.COND_RUNS RN ";
    result += "where ";
    result += "DS.CONDITION_DATA_SET_ID=DAT.CONDITION_DATA_SET_ID and ";
    result += "DS.CHANNEL_MAP_ID = CH.CHANNEL_MAP_ID and ";
    result += "KOC.KIND_OF_CONDITION_ID = DS.KIND_OF_CONDITION_ID and ";
    result += "RN.COND_RUN_ID=DS.COND_RUN_ID and ";
    result += "KOC.IS_RECORD_DELETED='F' and ";
    result += "DS.IS_RECORD_DELETED='F' and ";
    result += "KOC.NAME='HCAL Gains' and ";
    result += "RN.RUN_NAME='" + fTag + "'";
    return result;
  }
}

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

bool HcalDbOnline::getObject (HcalPedestals* fObject, const std::string& fTag) {
  return getObjectGeneric (fObject, fTag);
}

bool HcalDbOnline::getObject (HcalGains* fObject, const std::string& fTag) {
  return getObjectGeneric (fObject, fTag);
}

bool HcalDbOnline::getObject (HcalElectronicsMap* fObject, const std::string& fTag) {
  return false;
}


template <class T>
bool HcalDbOnline::getObjectGeneric (T* fObject, const std::string& fTag) {
  if (!fObject) return false;
  std::string sql_query = query (fObject, fTag);
  try {
    // std::cout << "executing query: \n" << sql_query << std::endl;
    //    oracle::occi::Statement* stmt = mConnect->createStatement ();
    mStatement->setPrefetchRowCount (100);
    mStatement->setSQL (sql_query);
    oracle::occi::ResultSet* rset = mStatement->executeQuery ();
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

//       std::cout << "getting data: " <<  value [0] << '/' <<  value [1] << '/' <<  value [2] << '/' <<  value [3]
//  		<< '/' <<  z << '/' <<  eta << '/' <<  phi << '/' <<  depth << '/' <<  subdet << std::endl;
      HcalSubdetector sub = subdet == "HB" ? HcalBarrel : 
	subdet == "HE" ? HcalEndcap :
	subdet == "HO" ? HcalOuter :
	subdet == "HF" ? HcalForward :  HcalSubdetector (0);
      HcalDetId id (sub, z * eta, phi, depth);
      fObject->addValue (id, value);
    }
    delete rset;
    //    delete stmt;
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbOnline::getObject exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  fObject->sort ();
  return true;
}  

