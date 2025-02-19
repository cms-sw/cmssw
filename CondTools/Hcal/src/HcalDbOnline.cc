
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbOnline.cc,v 1.22 2011/10/13 13:22:44 eulisse Exp $
//
#include <limits>
#include <string>
#include <iostream>
#include <sstream>

#include "OnlineDB/Oracle/interface/Oracle.h" 

#include "FWCore/Utilities/interface/Exception.h"
#include "CondTools/Hcal/interface/HcalDbOnline.h"

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

bool HcalDbOnline::getObject (HcalElectronicsMap* fObject, const std::string& fTag, IOVTime fTime) {
  if (!fObject) return false;
  std::string tag = fTag;
  if (tag.empty()) {
    tag = "9";
    std::cout << "HcalDbOnline::getObject (HcalElectronicsMap*..)-> Using default tag: " << tag << std::endl;
  }
  std::string sql_query ("");
  sql_query += "SELECT\n";         
  sql_query += " DAT2.SIDE, DAT2.ETA, DAT2.PHI, DAT2.DEPTH, DAT2.SUBDETECTOR,\n" ;
  sql_query += " DAT2.CRATE, DAT2.HTR_SLOT, DAT2.HTR_FPGA, DAT2.DCC, DAT2.DCC_SPIGOT, DAT2.HTR_FIBER, DAT2.FIBER_CHANNEL \n" ;
  sql_query += " FROM\n";         
  sql_query += " CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS DAT2,\n";
  sql_query += " CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS2,\n";
  sql_query += " CMS_HCL_CORE_CONDITION_OWNER.KINDS_OF_CONDITIONS KOC2\n";
  sql_query += " WHERE\n";
  sql_query += " DS2.CONDITION_DATA_SET_ID=DAT2.CONDITION_DATA_SET_ID\n";
  sql_query += " AND KOC2.KIND_OF_CONDITION_ID=DS2.KIND_OF_CONDITION_ID \n";
  sql_query += " AND KOC2.IS_RECORD_DELETED='F' AND DS2.IS_RECORD_DELETED='F' \n";
  sql_query += " AND KOC2.EXTENSION_TABLE_NAME='HCAL_HARDWARE_LOGICAL_MAPS' \n";
  sql_query += " AND DS2.VERSION='" + tag + "'\n";
  try {
    if (mVerbose) std::cout << "executing query: \n" << sql_query << std::endl;
    //    oracle::occi::Statement* stmt = mConnect->createStatement ();
    mStatement->setPrefetchRowCount (100);
    mStatement->setSQL (sql_query);
    oracle::occi::ResultSet* rset = mStatement->executeQuery ();
    while (rset->next ()) {
      int index = 1;
      int z = rset->getInt (index++) > 0 ? 1 : -1;
      int eta = rset->getInt (index++);
      int phi = rset->getInt (index++);
      int depth = rset->getInt (index++);
      std::string subdet = rset->getString (index++);
      int crate = rset->getInt (index++);
      int slot = rset->getInt (index++);
      int fpga = rset->getInt (index++) > 0 ? 1 : 0;
      int dcc = rset->getInt (index++);
      int spigot = rset->getInt (index++);
      int fiber = rset->getInt (index++);
      int fiberChannel = rset->getInt (index++);

      HcalElectronicsId eid (fiberChannel, fiber, spigot, dcc);
      eid.setHTR (crate, slot, fpga);

      HcalSubdetector sub = hcalSubdet (subdet);
      HcalDetId id (sub, z * eta, phi, depth);

//      fObject->setMapping (id, eid, HcalTrigTowerDetId ());
      DetId detid(id);
      fObject->mapEId2chId (eid,detid);
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

bool HcalDbOnline::getObject (HcalQIEData* fObject, const std::string& fTag, IOVTime fTime) {
  if (!fObject) return false;
  std::string sql_what ("");
  sql_what += " ADC_CH.SIDE, ADC_CH.ETA, ADC_CH.PHI, ADC_CH.DEPTH, ADC_CH.SUBDETECTOR,\n" ;
  sql_what += " DAT.CAP0_RANGE0_OFFSET, DAT.CAP0_RANGE0_SLOPE,\n" ;
  sql_what += " DAT.CAP0_RANGE1_OFFSET, DAT.CAP1_RANGE0_SLOPE,\n" ;
  sql_what += " DAT.CAP0_RANGE2_OFFSET, DAT.CAP2_RANGE0_SLOPE,\n" ;
  sql_what += " DAT.CAP0_RANGE3_OFFSET, DAT.CAP3_RANGE0_SLOPE,\n" ;
  sql_what += " DAT.CAP1_RANGE0_OFFSET, DAT.CAP0_RANGE1_SLOPE,\n" ;
  sql_what += " DAT.CAP1_RANGE1_OFFSET, DAT.CAP1_RANGE1_SLOPE,\n" ;
  sql_what += " DAT.CAP1_RANGE2_OFFSET, DAT.CAP2_RANGE1_SLOPE,\n" ;
  sql_what += " DAT.CAP1_RANGE3_OFFSET, DAT.CAP3_RANGE1_SLOPE,\n" ;
  sql_what += " DAT.CAP2_RANGE0_OFFSET, DAT.CAP0_RANGE2_SLOPE,\n" ;
  sql_what += " DAT.CAP2_RANGE1_OFFSET, DAT.CAP1_RANGE2_SLOPE,\n" ;
  sql_what += " DAT.CAP2_RANGE2_OFFSET, DAT.CAP2_RANGE2_SLOPE,\n" ;
  sql_what += " DAT.CAP2_RANGE3_OFFSET, DAT.CAP3_RANGE2_SLOPE,\n" ;
  sql_what += " DAT.CAP3_RANGE0_OFFSET, DAT.CAP0_RANGE3_SLOPE,\n" ;
  sql_what += " DAT.CAP3_RANGE1_OFFSET, DAT.CAP1_RANGE3_SLOPE,\n" ;
  sql_what += " DAT.CAP3_RANGE2_OFFSET, DAT.CAP2_RANGE3_SLOPE,\n" ;
  sql_what += " DAT.CAP3_RANGE3_OFFSET, DAT.CAP3_RANGE3_SLOPE \n" ;

  // HB/HE/HO
  std::string sql_hbheho ("");
  sql_hbheho += " FROM  \n";
  sql_hbheho += "  CMS_HCL_HCAL_CONDITION_OWNER.QIECARD_ADC_NORMMODE DAT\n";
  sql_hbheho += "  inner join CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS\n";
  sql_hbheho += "  on DAT.CONDITION_DATA_SET_ID=DS.CONDITION_DATA_SET_ID\n";
  sql_hbheho += "  inner join     \n";
  sql_hbheho += "  (  \n";
  sql_hbheho += "  select\n";
  sql_hbheho += "       LOGMAP.SIDE,\n";
  sql_hbheho += "       LOGMAP.ETA,\n";
  sql_hbheho += "       LOGMAP.PHI,\n";
  sql_hbheho += "       LOGMAP.DEPTH,\n";
  sql_hbheho += "       LOGMAP.SUBDETECTOR,\n";
  sql_hbheho += "       ADC_CHAIN.ADC_PART_ID\n";
  sql_hbheho += "      from\n";
  sql_hbheho += "       CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V2 LOGMAP\n";
  sql_hbheho += "       inner join CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS LOGMAP_DS\n";
  sql_hbheho += "       on LOGMAP_DS.CONDITION_DATA_SET_ID=LOGMAP.CONDITION_DATA_SET_ID\n";
  sql_hbheho += "       inner join\n";
  sql_hbheho += "       (\n";
  sql_hbheho += "            select ADC.PART_ID as ADC_PART_ID, \n";
  sql_hbheho += "            SLOT.NAME_LABEL    as RBX_SLOT,\n";
  sql_hbheho += "            RM.RM_SLOT         as RM_SLOT,\n";
  sql_hbheho += "            QIE.QIE_SLOT       as QIE_SLOT,\n";
  sql_hbheho += "            ADC.ADC_POSITION   as ADC\n";
  sql_hbheho += "            from CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTBOX_SLOTS SLOT\n";
  sql_hbheho += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTBOXS RBX\n";
  sql_hbheho += "            on RBX.PART_PARENT_ID=SLOT.PART_ID\n";
  sql_hbheho += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTMODULES RM\n";
  sql_hbheho += "            on RM.PART_PARENT_ID=RBX.PART_ID\n";
  sql_hbheho += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_QIECARDS QIE\n";
  sql_hbheho += "            on QIE.PART_PARENT_ID=RM.PART_ID\n";
  sql_hbheho += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_ADCS ADC\n";
  sql_hbheho += "            on ADC.PART_PARENT_ID=QIE.PART_ID \n";
  sql_hbheho += "       ) ADC_CHAIN\n";
  sql_hbheho += "       on ADC_CHAIN.RBX_SLOT = LOGMAP.RBX_SLOT and\n";
  sql_hbheho += "          ADC_CHAIN.RM_SLOT = LOGMAP.RM_SLOT and\n";
  sql_hbheho += "          ADC_CHAIN.QIE_SLOT = LOGMAP.QIE_SLOT and\n";
  sql_hbheho += "          ADC_CHAIN.ADC = LOGMAP.ADC\n";
  sql_hbheho += "          where LOGMAP_DS.VERSION='15'\n";
  sql_hbheho += "      ) ADC_CH\n";
  sql_hbheho += "      on DS.PART_ID=ADC_CH.ADC_PART_ID\n";
  sql_hbheho += "    where DS.VERSION='3'\n";


  // HF
  std::string sql_hf ("");
  sql_hf += " FROM  \n";
  sql_hf += "  CMS_HCL_HCAL_CONDITION_OWNER.QIECARD_ADC_NORMMODE DAT\n";
  sql_hf += "  inner join CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS\n";
  sql_hf += "  on DAT.CONDITION_DATA_SET_ID=DS.CONDITION_DATA_SET_ID\n";
  sql_hf += "  inner join     \n";
  sql_hf += "  (  \n";
  sql_hf += "  select\n";
  sql_hf += "       LOGMAP.SIDE,\n";
  sql_hf += "       LOGMAP.ETA,\n";
  sql_hf += "       LOGMAP.PHI,\n";
  sql_hf += "       LOGMAP.DEPTH,\n";
  sql_hf += "       LOGMAP.SUBDETECTOR,\n";
  sql_hf += "       ADC_CHAIN.ADC_PART_ID\n";
  sql_hf += "      from\n";
  sql_hf += "       CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V2 LOGMAP\n";
  sql_hf += "       inner join CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS LOGMAP_DS\n";
  sql_hf += "       on LOGMAP_DS.CONDITION_DATA_SET_ID=LOGMAP.CONDITION_DATA_SET_ID\n";
  sql_hf += "       inner join\n";
  sql_hf += "       (\n";
  sql_hf += "            select ADC.PART_ID as ADC_PART_ID, \n";
  sql_hf += "            SLOT.NAME_LABEL      as CRATE_SLOT,\n";
  sql_hf += "            QIE.QIE_SLOT       as QIE_SLOT,\n";
  sql_hf += "            ADC.ADC_POSITION   as ADC\n";
  sql_hf += "            from CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_WEDGE_SLOTS SLOT\n";
  sql_hf += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_HF_FE_CRATE CR\n";
  sql_hf += "            on CR.PART_PARENT_ID=SLOT.PART_ID\n";
  sql_hf += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_QIECARDS QIE\n";
  sql_hf += "            on QIE.PART_PARENT_ID=CR.PART_ID\n";
  sql_hf += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_ADCS ADC\n";
  sql_hf += "            on ADC.PART_PARENT_ID=QIE.PART_ID \n";
  sql_hf += "         ) ADC_CHAIN\n";
  sql_hf += "       on ADC_CHAIN.CRATE_SLOT = LOGMAP.RBX_SLOT and\n";
  sql_hf += "          ADC_CHAIN.QIE_SLOT = LOGMAP.QIE_SLOT and\n";
  sql_hf += "          ADC_CHAIN.ADC = LOGMAP.ADC\n";
  sql_hf += "          where LOGMAP_DS.VERSION='15'\n";
  sql_hf += "      ) ADC_CH\n";
  sql_hf += "      on DS.PART_ID=ADC_CH.ADC_PART_ID\n";
  sql_hf += "    where DS.VERSION='3'\n";
  
  std::string sql_query [2];
  sql_query [0] = " SELECT \n" + sql_what + sql_hbheho;
  sql_query [1] = " SELECT \n" + sql_what + sql_hf;

  try {
    for (int i = 0; i < 2; i++) {
      if (mVerbose) std::cout << "executing query: \n" << sql_query[i] << std::endl;
      mStatement->setPrefetchRowCount (100);
      mStatement->setSQL (sql_query [i]);
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
	
	HcalSubdetector sub = hcalSubdet (subdet);
	HcalDetId id (sub, z * eta, phi, depth);

	HcalQIECoder coder(id.rawId());
	for (int capId = 0; capId < 4; capId++) {
	  for (int range = 0; range < 4; range++) {
	    coder.setOffset (capId, range, offset [capId][range]);
	    coder.setSlope (capId, range, slope [capId][range]);	   
	  }
	}
	
	fObject->addCoder (coder);
      }
      delete rset;
    }
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbOnline::getObject exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  fObject->sort ();
  return true;
}

bool HcalDbOnline::getObject (HcalCalibrationQIEData* fObject, const std::string& fTag, IOVTime fTime) {
  if (!fObject) return false;
  std::string sql_what ("");
  sql_what += " ADC_CH.SIDE, ADC_CH.ETA, ADC_CH.PHI, ADC_CH.DEPTH, ADC_CH.SUBDETECTOR,\n" ;
  sql_what += " DAT.BIN0, DAT.BIN1, DAT.BIN2, DAT.BIN3, DAT.BIN4, DAT.BIN5, DAT.BIN6, DAT.BIN7,\n";
  sql_what += " DAT.BIN8, DAT.BIN9, DAT.BIN10, DAT.BIN11, DAT.BIN12, DAT.BIN13, DAT.BIN14, DAT.BIN15,\n";
  sql_what += " DAT.BIN16, DAT.BIN17, DAT.BIN18, DAT.BIN19, DAT.BIN20, DAT.BIN21, DAT.BIN22, DAT.BIN23,\n";
  sql_what += " DAT.BIN24, DAT.BIN25, DAT.BIN26, DAT.BIN27, DAT.BIN28, DAT.BIN29, DAT.BIN30, DAT.BIN31 \n";

  // HB/HE/HO
  std::string sql_hbheho ("");
  sql_hbheho += " FROM  \n";
  sql_hbheho += "  CMS_HCL_HCAL_CONDITION_OWNER.QIECARD_ADC_CALIBMODE DAT\n";
  sql_hbheho += "  inner join CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS\n";
  sql_hbheho += "  on DAT.CONDITION_DATA_SET_ID=DS.CONDITION_DATA_SET_ID\n";
  sql_hbheho += "  inner join     \n";
  sql_hbheho += "  (  \n";
  sql_hbheho += "  select\n";
  sql_hbheho += "       LOGMAP.SIDE,\n";
  sql_hbheho += "       LOGMAP.ETA,\n";
  sql_hbheho += "       LOGMAP.PHI,\n";
  sql_hbheho += "       LOGMAP.DEPTH,\n";
  sql_hbheho += "       LOGMAP.SUBDETECTOR,\n";
  sql_hbheho += "       ADC_CHAIN.ADC_PART_ID\n";
  sql_hbheho += "      from\n";
  sql_hbheho += "       CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V2 LOGMAP\n";
  sql_hbheho += "       inner join CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS LOGMAP_DS\n";
  sql_hbheho += "       on LOGMAP_DS.CONDITION_DATA_SET_ID=LOGMAP.CONDITION_DATA_SET_ID\n";
  sql_hbheho += "       inner join\n";
  sql_hbheho += "       (\n";
  sql_hbheho += "            select ADC.PART_ID as ADC_PART_ID, \n";
  sql_hbheho += "            SLOT.NAME_LABEL    as RBX_SLOT,\n";
  sql_hbheho += "            RM.RM_SLOT         as RM_SLOT,\n";
  sql_hbheho += "            QIE.QIE_SLOT       as QIE_SLOT,\n";
  sql_hbheho += "            ADC.ADC_POSITION   as ADC\n";
  sql_hbheho += "            from CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTBOX_SLOTS SLOT\n";
  sql_hbheho += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTBOXS RBX\n";
  sql_hbheho += "            on RBX.PART_PARENT_ID=SLOT.PART_ID\n";
  sql_hbheho += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_READOUTMODULES RM\n";
  sql_hbheho += "            on RM.PART_PARENT_ID=RBX.PART_ID\n";
  sql_hbheho += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_QIECARDS QIE\n";
  sql_hbheho += "            on QIE.PART_PARENT_ID=RM.PART_ID\n";
  sql_hbheho += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_ADCS ADC\n";
  sql_hbheho += "            on ADC.PART_PARENT_ID=QIE.PART_ID \n";
  sql_hbheho += "       ) ADC_CHAIN\n";
  sql_hbheho += "       on ADC_CHAIN.RBX_SLOT = LOGMAP.RBX_SLOT and\n";
  sql_hbheho += "          ADC_CHAIN.RM_SLOT = LOGMAP.RM_SLOT and\n";
  sql_hbheho += "          ADC_CHAIN.QIE_SLOT = LOGMAP.QIE_SLOT and\n";
  sql_hbheho += "          ADC_CHAIN.ADC = LOGMAP.ADC\n";
  sql_hbheho += "          where LOGMAP_DS.VERSION='15'\n";
  sql_hbheho += "      ) ADC_CH\n";
  sql_hbheho += "      on DS.PART_ID=ADC_CH.ADC_PART_ID\n";
  sql_hbheho += "    where DS.VERSION='3'\n";


  // HF
  std::string sql_hf ("");
  sql_hf += " FROM  \n";
  sql_hf += "  CMS_HCL_HCAL_CONDITION_OWNER.QIECARD_ADC_CALIBMODE DAT\n";
  sql_hf += "  inner join CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS DS\n";
  sql_hf += "  on DAT.CONDITION_DATA_SET_ID=DS.CONDITION_DATA_SET_ID\n";
  sql_hf += "  inner join     \n";
  sql_hf += "  (  \n";
  sql_hf += "  select\n";
  sql_hf += "       LOGMAP.SIDE,\n";
  sql_hf += "       LOGMAP.ETA,\n";
  sql_hf += "       LOGMAP.PHI,\n";
  sql_hf += "       LOGMAP.DEPTH,\n";
  sql_hf += "       LOGMAP.SUBDETECTOR,\n";
  sql_hf += "       ADC_CHAIN.ADC_PART_ID\n";
  sql_hf += "      from\n";
  sql_hf += "       CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V2 LOGMAP\n";
  sql_hf += "       inner join CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS LOGMAP_DS\n";
  sql_hf += "       on LOGMAP_DS.CONDITION_DATA_SET_ID=LOGMAP.CONDITION_DATA_SET_ID\n";
  sql_hf += "       inner join\n";
  sql_hf += "       (\n";
  sql_hf += "            select ADC.PART_ID as ADC_PART_ID, \n";
  sql_hf += "            SLOT.NAME_LABEL      as CRATE_SLOT,\n";
  sql_hf += "            QIE.QIE_SLOT       as QIE_SLOT,\n";
  sql_hf += "            ADC.ADC_POSITION   as ADC\n";
  sql_hf += "            from CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_WEDGE_SLOTS SLOT\n";
  sql_hf += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_HF_FE_CRATE CR\n";
  sql_hf += "            on CR.PART_PARENT_ID=SLOT.PART_ID\n";
  sql_hf += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_QIECARDS QIE\n";
  sql_hf += "            on QIE.PART_PARENT_ID=CR.PART_ID\n";
  sql_hf += "            inner join CMS_HCL_CORE_CONSTRUCT_OWNER.V_HCAL_ADCS ADC\n";
  sql_hf += "            on ADC.PART_PARENT_ID=QIE.PART_ID \n";
  sql_hf += "         ) ADC_CHAIN\n";
  sql_hf += "       on ADC_CHAIN.CRATE_SLOT = LOGMAP.RBX_SLOT and\n";
  sql_hf += "          ADC_CHAIN.QIE_SLOT = LOGMAP.QIE_SLOT and\n";
  sql_hf += "          ADC_CHAIN.ADC = LOGMAP.ADC\n";
  sql_hf += "          where LOGMAP_DS.VERSION='15'\n";
  sql_hf += "      ) ADC_CH\n";
  sql_hf += "      on DS.PART_ID=ADC_CH.ADC_PART_ID\n";
  sql_hf += "    where DS.VERSION='3'\n";

  
  std::string sql_query [2];
  sql_query [0] = " SELECT \n" + sql_what + sql_hbheho;
  sql_query [1] = " SELECT \n" + sql_what + sql_hf;

  try {
    for (int i = 0; i < 2; i++) {
      if (mVerbose) std::cout << "executing query: \n" << sql_query [i] << std::endl;
      //    oracle::occi::Statement* stmt = mConnect->createStatement ();
      mStatement->setPrefetchRowCount (100);
      mStatement->setSQL (sql_query [i]);
      oracle::occi::ResultSet* rset = mStatement->executeQuery ();
      while (rset->next ()) {
	int index = 1;
	int z = rset->getInt (index++);
	int eta = rset->getInt (index++);
	int phi = rset->getInt (index++);
	int depth = rset->getInt (index++);
	std::string subdet = rset->getString (index++);
	float values [32];
	for (unsigned bin = 0; bin < 32; bin++) values [bin] = rset->getFloat (index++);
	
	HcalSubdetector sub = hcalSubdet (subdet);
	HcalDetId id (sub, z * eta, phi, depth);

	HcalCalibrationQIECoder coder(id.rawId());
	coder.setMinCharges (values);
	
	fObject->addCoder (coder);
      }
      delete rset;
    }
    //    delete stmt;
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbOnline::getObject exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  fObject->sort ();
  return true;
}


bool HcalDbOnline::getObject (HcalPedestals* fObject, HcalPedestalWidths* fWidths, const std::string& fTag, IOVTime fTime) {
  if (!fObject && !fWidths) return false;
  std::ostringstream sTime;
  sTime << fTime;
  std::string sql_query ("");
  sql_query += "SELECT\n"; 
  sql_query += " Z, ETA, PHI, DEPTH, DETECTOR_NAME\n"; 
  sql_query += " , CAPACITOR_0_VALUE, CAPACITOR_1_VALUE, CAPACITOR_2_VALUE, CAPACITOR_3_VALUE\n"; 
  sql_query += " , SIGMA_0_0, SIGMA_0_1, SIGMA_0_2, SIGMA_0_3, SIGMA_1_1, SIGMA_1_2, SIGMA_1_3, SIGMA_2_2, SIGMA_2_3, SIGMA_3_3\n"; 
  sql_query += " , RUN_NUMBER, INTERVAL_OF_VALIDITY_BEGIN, INTERVAL_OF_VALIDITY_END\n"; 
  sql_query += "FROM V_HCAL_PEDESTALS_V2\n"; 
  sql_query += "WHERE TAG_NAME='" + fTag + "'\n";
  sql_query += "AND INTERVAL_OF_VALIDITY_BEGIN=" + sTime.str() + "\n";
  sql_query += "AND (INTERVAL_OF_VALIDITY_END IS NULL OR INTERVAL_OF_VALIDITY_END>" + sTime.str() + ")\n";
  try {
    if (mVerbose) std::cout << "executing query: \n" << sql_query << std::endl;
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

      float values [4];
      float widths [4][4];
      for (int i = 0; i < 4; i++) values[i] = rset->getFloat (index++);
      for (int i = 0; i < 4; i++) 
	for (int j = i; j < 4; j++) widths [i][j] = rset->getFloat (index++);

//       unsigned long run = rset->getNumber (index++);
//       unsigned long iovBegin = rset->getNumber (index++);
//       unsigned long iovEnd = rset->getNumber (index++);

      HcalSubdetector sub = hcalSubdet (subdet);
      HcalDetId id (sub, z * eta, phi, depth);
      
      if (fObject) {
	  if (fObject->exists(id) )
	    std::cerr << "HcalDbOnline::getObject-> Ignore data to redefine channel " << id.rawId() << std::endl;
	  else
	    {
	      HcalPedestal myped(id,values[0],values[1],values[2],values[3]);
	      fObject->addValues(myped);
	    }
      }
      if (fWidths) {
	if (fWidths->exists(id) )
	  std::cerr << "HcalDbOnline::getObject-> Ignore data to redefine channel " << id.rawId() << std::endl;
	else
	  {
	    HcalPedestalWidth mywidth(id);
	    for (int i = 0; i < 4; i++) 
	      for (int j = i; j < 4; j++) mywidth.setSigma (i, j, widths [i][j]);
	    fWidths->addValues(mywidth);
	  }
      }
    }
    delete rset;
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbOnline::getObject exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  //  if (fObject) fObject->sort ();
  //  if (fWidths) fWidths->sort ();
  return true;
}

bool HcalDbOnline::getObject (HcalPedestals* fObject, const std::string& fTag, IOVTime fTime) {
  return getObject (fObject, (HcalPedestalWidths*)0, fTag, fTime);
}

bool HcalDbOnline::getObject (HcalPedestalWidths* fObject, const std::string& fTag, IOVTime fTime) {
  return getObject ((HcalPedestals*)0, fObject, fTag, fTime);
}

bool HcalDbOnline::getObject (HcalGains* fObject, HcalGainWidths* fWidths, const std::string& fTag, IOVTime fTime) {
  if (!fObject && !fWidths) return false;
  std::ostringstream sTime;
  sTime << fTime;
  std::string sql_query ("");
  sql_query += "SELECT\n"; 
  sql_query += " Z, ETA, PHI, DEPTH, DETECTOR_NAME\n"; 
  sql_query += " , CAPACITOR_0_VALUE, CAPACITOR_1_VALUE, CAPACITOR_2_VALUE, CAPACITOR_3_VALUE\n"; 
  sql_query += " , CAPACITOR_0_ERROR, CAPACITOR_1_ERROR, CAPACITOR_2_ERROR, CAPACITOR_3_ERROR\n"; 
  sql_query += " , RUN_NUMBER, INTERVAL_OF_VALIDITY_BEGIN, INTERVAL_OF_VALIDITY_END\n"; 
  sql_query += "FROM V_HCAL_GAIN_CALIBRATIONS\n"; 
  sql_query += "WHERE TAG_NAME='" + fTag + "'\n";
  sql_query += "AND INTERVAL_OF_VALIDITY_BEGIN=" + sTime.str() + "\n";
  sql_query += "AND (INTERVAL_OF_VALIDITY_END IS NULL OR INTERVAL_OF_VALIDITY_END>" + sTime.str() + ")\n";
  try {
    if (mVerbose) std::cout << "executing query: \n" << sql_query << std::endl;
    mStatement->setPrefetchRowCount (100);
    mStatement->setSQL (sql_query);
    oracle::occi::ResultSet* rset = mStatement->executeQuery ();
    std::cout << "query is executed... " << std::endl;
    while (rset->next ()) {
      int index = 1;
      int z = rset->getInt (index++);
      int eta = rset->getInt (index++);
      int phi = rset->getInt (index++);
      int depth = rset->getInt (index++);
      std::string subdet = rset->getString (index++);

      float values [4];
      for (int i = 0; i < 4; i++) values[i] = rset->getFloat (index++);
      for (int i = 0; i < 4; i++) rset->getFloat (index++);
//       unsigned long run = rset->getNumber (index++);
//       unsigned long iovBegin = rset->getNumber (index++);
//       unsigned long iovEnd = rset->getNumber (index++);

      HcalSubdetector sub = hcalSubdet (subdet);
      HcalDetId id (sub, z * eta, phi, depth);

      if (fObject) {
	if (fObject->exists(id) )
	  std::cerr << "HcalDbOnline::getObject-> Ignore data to redefine channel " << id.rawId() << std::endl;
	else
	  {
	    HcalGain mygain(id,values[0],values[1],values[2],values[3]);
	    fObject->addValues (mygain);
	  }
      }
      if (fWidths) {
	if (fWidths->exists(id) )
	  std::cerr << "HcalDbOnline::getObject-> Ignore data to redefine channel " << id.rawId() << std::endl;
	else
	  {
	    HcalGainWidth mywid(id,values[0],values[1],values[2],values[3]);
	    fWidths->addValues(mywid);
	  }
      }
    }
    delete rset;
  }
  catch (oracle::occi::SQLException& sqlExcp) {
    std::cerr << "HcalDbOnline::getObject exception-> " << sqlExcp.getErrorCode () << ": " << sqlExcp.what () << std::endl;
  }
  //  if (fObject) fObject->sort ();
  //  if (fWidths) fWidths->sort ();
  return true;
}

bool HcalDbOnline::getObject (HcalGains* fObject, const std::string& fTag, IOVTime fTime) {
  return getObject (fObject, (HcalGainWidths*) 0, fTag, fTime);
}

bool HcalDbOnline::getObject (HcalGainWidths* fWidths, const std::string& fTag, IOVTime fTime) {
  return getObject ((HcalGains*) 0, fWidths, fTag, fTime);
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

