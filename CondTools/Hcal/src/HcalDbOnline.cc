
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbHardcode.cc,v 1.1 2005/12/15 23:39:36 fedor Exp $
//
#include <string>
#include <iostream>

#define OTL_ORA9I // Compile OTL 4.0/OCI9i
#include "otlv4.h" // include the OTL 4.0 header file

#include "CondTools/Hcal/interface/HcalDbOnline.h"


HcalDbOnline::HcalDbOnline (const std::string& fDb) 
: mConnect (0) {
  mConnect = new otl_connect;
  otl_connect::otl_initialize(); // initialize OCI environment
  try{
    mConnect->rlogon(fDb.c_str ()); // connect to Oracle
  }
  catch(otl_exception& p){ // intercept OTL exceptions
    std::cerr << p.msg << std::endl; // print out error message
    std::cerr << p.stm_text << std::endl; // print out SQL that caused the error
    std::cerr << p.var_info << std::endl; // print out the variable that caused the error
  }
}

HcalDbOnline::~HcalDbOnline () {
  if (mConnect) mConnect->logoff(); // disconnect from Oracle
  delete mConnect;
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
  
  float value [4];
  int z, eta, phi, depth;
  char subdet_c [128];
  
  try {
    std::cout << "executing query: \n" << query << std::endl;
    otl_stream input (500, query.c_str (), *mConnect); 
    while (!input.eof ()) {
      input >> value [0] >> value [1] >> value [2] >> value [3];
      input >> z >> eta >> phi >> depth >> subdet_c;
//       std::cout << "getting data: " <<  value [0] << '/' <<  value [1] << '/' <<  value [2] << '/' <<  value [3]
// 		<< '/' <<  z << '/' <<  eta << '/' <<  phi << '/' <<  depth << '/' <<  subdet_c << std::endl;
      std::string subdet (subdet_c);
      HcalSubdetector sub = subdet == "HB" ? HcalBarrel : 
	subdet == "HE" ? HcalEndcap :
	subdet == "HO" ? HcalOuter :
	subdet == "HF" ? HcalForward :  HcalSubdetector (0);
      HcalDetId id (sub, z * eta, phi, depth);
      result->addValue (id, value);
    }
  }
  catch(otl_exception& p){ // intercept OTL exceptions
    std::cerr << "HcalDbOnline::getPedestals->" << p.msg << std::endl; // print out error message
    std::cerr << p.stm_text << std::endl; // print out SQL that caused the error
    std::cerr << p.var_info << std::endl; // print out the variable that caused the error
  }
  result->sort ();
  return result;
}  

std::auto_ptr <HcalGains> HcalDbOnline::getGains (const std::string& fTag) {
  std::auto_ptr <HcalGains> result (new HcalGains ());
  return result;
}
