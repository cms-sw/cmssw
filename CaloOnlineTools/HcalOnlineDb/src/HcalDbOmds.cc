// -*- C++ -*-
//
// Original Author:  Gena Kukartsev Mar 11, 2009
// Adapted from HcalDbASCIIIO.cc,v 1.41
// $Id: HcalDbOmds.cc,v 1.5 2009/03/24 14:33:28 kukartse Exp $
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


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalPedestals* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalPedestalWidths* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalGains* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalGainWidths* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalQIEData* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalCalibrationQIEData* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalElectronicsMap* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalChannelQuality* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalRespCorrs* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}

// Oracle database connection ownership is transferred here, DO terminate after use
bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalZSThresholds* fObject) {
  bool result=true;
  if (!fObject) fObject = new HcalZSThresholds;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setString(2,fVersion);
    //stmt->setInt(3,fSubversion);

    ResultSet *rs = stmt->executeQuery();

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
  return result;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const std::string & fQuery,
			    HcalL1TriggerObjects* fObject) {
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
