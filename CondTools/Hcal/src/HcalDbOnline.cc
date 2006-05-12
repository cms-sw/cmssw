
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbOnline.cc,v 1.4 2006/02/08 20:25:55 fedor Exp $
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

bool HcalDbOnline::getObject (HcalQIEData* fObject, const std::string& fTag) {
  if (!fObject) return false;
  std::string sql_query ("");
  sql_query += "SELECT";         
  sql_query += " DAT2.SIDE, DAT2.ETA, DAT2.PHI, DAT2.DEPTH, DAT2.SUBDETECTOR," ;
  sql_query += " DAT.CAP0_RANGE0_OFFSET, DAT.CAP0_RANGE0_SLOPE," ;
  sql_query += " DAT.CAP0_RANGE1_OFFSET, DAT.CAP1_RANGE0_SLOPE," ;
  sql_query += " DAT.CAP0_RANGE2_OFFSET, DAT.CAP2_RANGE0_SLOPE," ;
  sql_query += " DAT.CAP0_RANGE3_OFFSET, DAT.CAP3_RANGE0_SLOPE," ;
  sql_query += " DAT.CAP1_RANGE0_OFFSET, DAT.CAP0_RANGE1_SLOPE," ;
  sql_query += " DAT.CAP1_RANGE1_OFFSET, DAT.CAP1_RANGE1_SLOPE," ;
  sql_query += " DAT.CAP1_RANGE2_OFFSET, DAT.CAP2_RANGE1_SLOPE," ;
  sql_query += " DAT.CAP1_RANGE3_OFFSET, DAT.CAP3_RANGE1_SLOPE," ;
  sql_query += " DAT.CAP2_RANGE0_OFFSET, DAT.CAP0_RANGE2_SLOPE," ;
  sql_query += " DAT.CAP2_RANGE1_OFFSET, DAT.CAP1_RANGE2_SLOPE," ;
  sql_query += " DAT.CAP2_RANGE2_OFFSET, DAT.CAP2_RANGE2_SLOPE," ;
  sql_query += " DAT.CAP2_RANGE3_OFFSET, DAT.CAP3_RANGE2_SLOPE," ;
  sql_query += " DAT.CAP3_RANGE0_OFFSET, DAT.CAP0_RANGE3_SLOPE," ;
  sql_query += " DAT.CAP3_RANGE1_OFFSET, DAT.CAP1_RANGE3_SLOPE," ;
  sql_query += " DAT.CAP3_RANGE2_OFFSET, DAT.CAP2_RANGE3_SLOPE," ;
  sql_query += " DAT.CAP3_RANGE3_OFFSET, DAT.CAP3_RANGE3_SLOPE," ;
  sql_query += " SLOT.NAME_LABEL, RM.RM_SLOT, QIE.QIE_SLOT, ADC.ADC_POSITION" ;
  sql_query += " FROM " ;
  sql_query += " CMS_HCL_HCAL_CONDITION_OWNER.QIECARD_ADC_NORMMODE DAT," ;
  sql_query += " CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS," ;
  sql_query += " CMS_HCL_CORE_CONDITION_OWNER.KINDS_OF_CONDITIONS KOC," ;
  sql_query += " CMS_HCL_CORE_CONDITION_OWNER.COND_RUNS RN,";
  sql_query += " CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_ADCS ADC,";
  sql_query += " CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_QIECARDS QIE,";
  sql_query += " CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTMODULES RM,";
  sql_query += " CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTBOXS RBX,";
  sql_query += " CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTBOX_SLOTS SLOT,";
  sql_query += " CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS DAT2,";
  sql_query += " CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS2,";
  sql_query += " CMS_HCL_CORE_CONDITION_OWNER.KINDS_OF_CONDITIONS KOC2,";
  sql_query += " CMS_HCL_CORE_CONDITION_OWNER.COND_RUNS RN2";
  sql_query += " WHERE";
  sql_query += " DS.CONDITION_DATA_SET_ID=DAT.CONDITION_DATA_SET_ID ";
  sql_query += " AND DS.PART_ID=ADC.PART_ID";
  sql_query += " AND ADC.PART_PARENT_ID=QIE.PART_ID";
  sql_query += " AND QIE.PART_PARENT_ID=RM.PART_ID";
  sql_query += " AND RM.PART_PARENT_ID=RBX.PART_ID";
  sql_query += " AND RBX.PART_PARENT_ID=SLOT.PART_ID";
  sql_query += " AND KOC.KIND_OF_CONDITION_ID = DS.KIND_OF_CONDITION_ID " ;       
  sql_query += " AND RN.COND_RUN_ID=DS.COND_RUN_ID";
  sql_query += " AND KOC.IS_RECORD_DELETED='F' AND DS.IS_RECORD_DELETED='F'";
  sql_query += " AND KOC.NAME='QIE Responce Normal Mode' and DS.VERSION='2'";
  sql_query += " AND";
  sql_query += " DS2.CONDITION_DATA_SET_ID=DAT2.CONDITION_DATA_SET_ID AND";
  sql_query += " KOC2.KIND_OF_CONDITION_ID=DS2.KIND_OF_CONDITION_ID AND";
  sql_query += " RN2.COND_RUN_ID=DS2.COND_RUN_ID AND";
  sql_query += " KOC2.IS_RECORD_DELETED='F' AND DS2.IS_RECORD_DELETED='F' AND";
  sql_query += " KOC2.EXTENSION_TABLE_NAME='HCAL_HARDWARE_LOGICAL_MAPS' AND";
  sql_query += " RN2.RUN_NAME='HCAL-LOGICAL-MAP-27APR06' AND";
  sql_query += " DS2.VERSION='5'";
  sql_query += " AND        ";
  sql_query += " SLOT.NAME_LABEL=DAT2.RBX_SLOT AND";
  sql_query += " RM.RM_SLOT=DAT2.RM_SLOT AND";
  sql_query += " QIE.QIE_SLOT=DAT2.QIE_SLOT AND";
  sql_query += " ADC.ADC_POSITION=DAT2.ADC";
  
  try {
    // std::cout << "executing query: \n" << sql_query << std::endl;
    //    oracle::occi::Statement* stmt = mConnect->createStatement ();
    mStatement->setPrefetchRowCount (100);
    mStatement->setSQL (sql_query);
    oracle::occi::ResultSet* rset = mStatement->executeQuery ();
    while (rset->next ()) {
      int index = 1;
      int z = rset->getInt (index++);
      int eta = rset->getInt (index++);
      int phi = rset->getInt (index++);
      int depth = rset->getInt (index++);
      std::string subdet = rset->getString (index++);
      float offset [4][4];
      float slope [4][4];
      for (int capId = 0; capId < 4; capId++) {
	for (int range = 0; range < 4; range++) {
	  offset [capId][range] = rset->getFloat (index++);
	  slope [capId][range] = rset->getFloat (index++);
	}
      }
      std::string slot = rset->getString (index++);
      int rm = rset->getInt (index++);
      int qie = rset->getInt (index++);
      int adc = rset->getInt (index++);

      HcalQIECoder coder;
      for (int capId = 0; capId < 4; capId++) {
	for (int range = 0; range < 4; range++) {
	  coder.setOffset (capId, range, offset [capId][range]);
	  coder.setSlope (capId, range, slope [capId][range]);	   
	}
      }

      HcalSubdetector sub = subdet == "HB" ? HcalBarrel : 
	subdet == "HE" ? HcalEndcap :
	subdet == "HO" ? HcalOuter :
	subdet == "HF" ? HcalForward :  HcalSubdetector (0);
      HcalDetId id (sub, z * eta, phi, depth);

      fObject->addCoder (id, coder);
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

