// -*- C++ -*-
//
// Original Author:  Gena Kukartsev Mar 11, 2009
// Adapted from HcalDbASCIIIO.cc,v 1.41
// $Id: HcalDbOmds.cc,v 1.3 2009/03/15 14:36:14 kukartse Exp $
//
//
#include <vector>
#include <string>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalDbOmds.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalPedestals* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalPedestalWidths* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalGains* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalGainWidths* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalQIEData* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalCalibrationQIEData* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalElectronicsMap* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalChannelQuality* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalRespCorrs* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}

// Oracle database connection ownership is transferred here, DO terminate after use
bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalZSThresholds* fObject) {
  bool result=true;
  if (!fObject) fObject = new HcalZSThresholds;
  try {
    Statement * stmt = connection->createStatement();

    std::string query = " SELECT zero_suppression,z*eta as ieta,phi,depth,detector_name as subdetector ";
    query            += " FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_ZERO_SUPPRESSION ";
    //query            += " FROM CMS_HCL_HCAL_COND.V_HCAL_ZERO_SUPPRESSION ";
    query            += " WHERE TAG_NAME='GREN_ZS_9adc_v2'";

    ResultSet *rs = stmt->executeQuery(query.c_str());

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    while (rs->next()) {
      _row.count();
      int zs = rs->getInt(1);
      int ieta = rs->getInt(2);
      int iphi = rs->getInt(3);
      int depth = rs->getInt(4);
      HcalSubdetector subdetector = get_subdetector(rs->getString(5));
      HcalDetId id(subdetector,ieta,iphi,depth);
      //cout << "DEBUG: " << id << " " << zs << endl;
      HcalZSThreshold * fCondObject = new HcalZSThreshold(id, zs);
      fObject->addValues(*fCondObject);
      delete fCondObject;
    }
    //Always terminate statement
    connection->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw cms::Exception("ReadError") << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
  }


  /*
  oracle::occi::ResultSet * rs=0;
  if (!fObject) fObject = new HcalZSThresholds;
  HcalDetId id(HcalBarrel,15,49,1);
  int zs=9;
  HcalZSThreshold * fCondObject = new HcalZSThreshold(id, zs);
  fObject->addValues(*fCondObject);
  delete fCondObject;
  */  

  return result;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, const std::string & fTag, HcalL1TriggerObjects* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::dumpObject (std::ostream& fOutput, const HcalZSThresholds& fObject) {
  return true;
}


HcalSubdetector HcalDbOmds::get_subdetector( string _det )
{
  HcalSubdetector result;
  if      ( _det.find("HB") != string::npos ) result = HcalBarrel;
  else if ( _det.find("HE") != string::npos ) result = HcalEndcap;
  else if ( _det.find("HF") != string::npos ) result = HcalForward;
  else if ( _det.find("HO") != string::npos ) result = HcalOuter;
  else                                        result = HcalOther;  

  return result;
}
