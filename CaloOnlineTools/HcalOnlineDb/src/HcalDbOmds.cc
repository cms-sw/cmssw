// -*- C++ -*-
//
// Original Author:  Gena Kukartsev Mar 11, 2009
// Adapted from HcalDbASCIIIO.cc,v 1.41
// $Id: HcalDbOmds.cc,v 1.21 2012/11/12 20:49:45 dlange Exp $
//
#include <vector>
#include <string>

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalDbOmds.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"

typedef oracle::occi::ResultSet ResultSet;
typedef oracle::occi::SQLException SQLException;

template<class T>
bool HcalDbOmds::from_string(T& t, const std::string& s, std::ios_base& (*f)(std::ios_base&)) {
  std::istringstream iss(s);
  return !(iss >> f >> t).fail();
}

// get the proper detId from the result of the oracle query
// assumed that channel info comes in from of the ResultSet
// in the following order (some may not be filled):
// 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
// 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
DetId HcalDbOmds::getId(oracle::occi::ResultSet * rs){
  std::string _name = rs->getString(1);
  //std::cerr << "DEBUG: name - " << _name << std::endl;
  if (rs->getString(1).find("HcalDetId")!=std::string::npos){
    //std::cerr << "DEBUG: HcalDetId" << std::endl;
    return HcalDetId(get_subdetector(rs->getString(2)),
		     rs->getInt(3),
		     rs->getInt(4),
		     rs->getInt(5));
  }
  else if (rs->getString(1).find("HcalCalibDetId")!=std::string::npos){
    //std::cerr << "DEBUG: HcalCalibDetId" << std::endl;
    return HcalCalibDetId(get_subdetector(rs->getString(2)),
			  rs->getInt(3),
			  rs->getInt(4),
			  rs->getInt(6));
  }
  else if (rs->getString(1).find("HcalTrigTowerDetId")!=std::string::npos){
    //std::cerr << "DEBUG: HcalTrigTowerDetId" << std::endl;
    return HcalTrigTowerDetId(rs->getInt(3),
			      rs->getInt(4));
  }
  else if (rs->getString(1).find("HcalZDCDetId")!=std::string::npos){
    //std::cerr << "DEBUG: HcalZDCDetId" << std::endl;
    return HcalZDCDetId(get_zdc_section(rs->getString(7)),
			rs->getInt(8)>0,
			rs->getInt(11));
  }
  else if (rs->getString(1).find("HcalCastorDetId")!=std::string::npos){
    //std::cerr << "DEBUG: HcalCastorDetId" << std::endl;
    return HcalCastorDetId(rs->getInt(8)>0,
			   rs->getInt(9)>0,
			   rs->getInt(10));
  }
  else return 0;
}

bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalPedestals* fObject) {
  bool result=true;
  if (!fObject) return false; // fObject = new HcalPedestals;  
  int _unit=0;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    //stmt->setString(2,fVersion);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=12; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12. is_adc_counts, 13-16. cap0-3, 17-20. variance0-3
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      _unit = rs->getInt(12);
      float cap0 = rs->getFloat(13);
      float cap1 = rs->getFloat(14);
      float cap2 = rs->getFloat(15);
      float cap3 = rs->getFloat(16);
      float variance0 = rs->getFloat(17);
      float variance1 = rs->getFloat(18);
      float variance2 = rs->getFloat(19);
      float variance3 = rs->getFloat(20);
      //int ieta = rs->getInt(21);
      //int iphi = rs->getInt(22);
      //int depth = rs->getInt(23);
      //HcalSubdetector subdetector = get_subdetector(rs->getString(24));
      //HcalDetId id(subdetector,ieta,iphi,depth);
      std::cout << "DEBUG: " << std::endl;
      //std::cout << "DEBUG: " << id << " " << cap0 << " " << cap1 << " " << cap2 << " " << cap3 << std::endl;
      HcalPedestal * fCondObject = new HcalPedestal(id.rawId(), cap0, cap1, cap2, cap3, variance0, variance1, variance2, variance3);
      fObject->addValues(*fCondObject);
      delete fCondObject;
    }
    //Always terminate statement
    connection->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw cms::Exception("ReadError") << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
  }
  bool unit_is_adc = false;
  if (_unit!=0) unit_is_adc = true;
  fObject->setUnitADC(unit_is_adc);
  return result;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalPedestalWidths* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalPedestalWidths;
  int _unit=0;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    rs->setMaxColumnSize(1,128);
    rs->setMaxColumnSize(2,128);
    rs->setMaxColumnSize(3,128);
    rs->setMaxColumnSize(4,128);
    rs->setMaxColumnSize(5,128);
    rs->setMaxColumnSize(6,128);
    rs->setMaxColumnSize(7,128);
    rs->setMaxColumnSize(8,128);
    rs->setMaxColumnSize(9,128);
    rs->setMaxColumnSize(10,128);
    rs->setMaxColumnSize(11,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12. is_ADC(int), 13-28. covariance00-01-...-10-11-...33
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      _unit = rs->getInt(12);
      float covariance_00 = rs->getFloat(13);
      float covariance_01 = rs->getFloat(14);
      float covariance_02 = rs->getFloat(15);
      float covariance_03 = rs->getFloat(16);
      float covariance_10 = rs->getFloat(17);
      float covariance_11 = rs->getFloat(18);
      float covariance_12 = rs->getFloat(19);
      float covariance_13 = rs->getFloat(20);
      float covariance_20 = rs->getFloat(21);
      float covariance_21 = rs->getFloat(22);
      float covariance_22 = rs->getFloat(23);
      float covariance_23 = rs->getFloat(24);
      float covariance_30 = rs->getFloat(25);
      float covariance_31 = rs->getFloat(26);
      float covariance_32 = rs->getFloat(27);
      float covariance_33 = rs->getFloat(28);
      HcalPedestalWidth * fCondObject = new HcalPedestalWidth(id);
      fCondObject->setSigma(0,0,covariance_00);
      fCondObject->setSigma(0,1,covariance_01);
      fCondObject->setSigma(0,2,covariance_02);
      fCondObject->setSigma(0,3,covariance_03);
      fCondObject->setSigma(1,0,covariance_10);
      fCondObject->setSigma(1,1,covariance_11);
      fCondObject->setSigma(1,2,covariance_12);
      fCondObject->setSigma(1,3,covariance_13);
      fCondObject->setSigma(2,0,covariance_20);
      fCondObject->setSigma(2,1,covariance_21);
      fCondObject->setSigma(2,2,covariance_22);
      fCondObject->setSigma(2,3,covariance_23);
      fCondObject->setSigma(3,0,covariance_30);
      fCondObject->setSigma(3,1,covariance_31);
      fCondObject->setSigma(3,2,covariance_32);
      fCondObject->setSigma(3,3,covariance_33);
      fObject->addValues(*fCondObject);
      delete fCondObject;
    }
    //Always terminate statement
    connection->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw cms::Exception("ReadError") << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
  }
  bool unit_is_adc = false;
  if (_unit!=0) unit_is_adc = true;
  fObject->setUnitADC(unit_is_adc);
  return result;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalGains* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalGains;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=12; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12-15. cap0-3
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      float cap0 = rs->getFloat(12);
      float cap1 = rs->getFloat(13);
      float cap2 = rs->getFloat(14);
      float cap3 = rs->getFloat(15);
      //int ieta = rs->getInt(5);
      //int iphi = rs->getInt(6);
      //int depth = rs->getInt(7);
      //HcalSubdetector subdetector = get_subdetector(rs->getString(8));
      //HcalDetId id(subdetector,ieta,iphi,depth);
      //std::cout << "DEBUG: " << id << " " << cap0 << " " << cap1 << " " << cap2 << " " << cap3 << std::endl;
      HcalGain * fCondObject = new HcalGain(id, cap0, cap1, cap2, cap3);
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
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalGainWidths* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalGainWidths;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=12; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12-15. cap0-3
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      float cap0 = rs->getFloat(12);
      float cap1 = rs->getFloat(13);
      float cap2 = rs->getFloat(14);
      float cap3 = rs->getFloat(15);
      //int ieta = rs->getInt(5);
      //int iphi = rs->getInt(6);
      //int depth = rs->getInt(7);
      //HcalSubdetector subdetector = get_subdetector(rs->getString(8));
      //HcalDetId id(subdetector,ieta,iphi,depth);
      //std::cout << "DEBUG: " << id << " " << cap0 << " " << cap1 << " " << cap2 << " " << cap3 << std::endl;
      HcalGainWidth * fCondObject = new HcalGainWidth(id, cap0, cap1, cap2, cap3);
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
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalQIEData* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalQIEData;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    rs->setMaxColumnSize(1,128);
    rs->setMaxColumnSize(2,128);
    rs->setMaxColumnSize(3,128);
    rs->setMaxColumnSize(4,128);
    rs->setMaxColumnSize(5,128);
    rs->setMaxColumnSize(6,128);
    rs->setMaxColumnSize(7,128);
    rs->setMaxColumnSize(8,128);
    rs->setMaxColumnSize(9,128);
    rs->setMaxColumnSize(10,128);
    rs->setMaxColumnSize(11,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 13-27. cap0_range0_slope, cap0_range1_slope... 33, 28-43. cap0_range0_offset, cap0_range1_offset...
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      fObject->sort();
      float items[32];
      for (int _i=0; _i!=32; _i++) items[_i] = rs->getFloat(_i+12);
      HcalQIECoder * fCondObject = new HcalQIECoder(id.rawId());
      for (unsigned capid = 0; capid < 4; capid++) {
	for (unsigned range = 0; range < 4; range++) {
	  fCondObject->setSlope (capid, range, items[capid*4+range]);
	}
      }
      for (unsigned capid = 0; capid < 4; capid++) {
	for (unsigned range = 0; range < 4; range++) {
	  fCondObject->setOffset (capid, range, items[16+capid*4+range]);
	}
      }
      fObject->addCoder(*fCondObject);
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
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalCalibrationQIEData* fObject) {
  std::cerr << "NOT IMPLEMENTED!" << std::endl;
  return false;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalElectronicsMap* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalElectronicsMap;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=20; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12. i, 13. crate, 14. slot, 15. tb, 16. dcc, 17. spigot, 18. fiber(slb), 19. fiberchan(slbchan)
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      std::string _obj_name = rs->getString(1);
      int crate = rs->getInt(13);
      int slot = rs->getInt(14);
      int dcc = rs->getInt(16);
      int spigot = rs->getInt(17);
      std::string tb = rs->getString(15);
      int top = 1;
      if (tb.find("b")!=std::string::npos) top = 0;
      HcalElectronicsId * fCondObject = 0;
      if (_obj_name.find("HcalTrigTowerDetId")!=std::string::npos){
	int slbCh = rs->getInt(19);
	int slb = rs->getInt(18);
	fCondObject = new HcalElectronicsId(slbCh, slb, spigot, dcc,crate,slot,top);
      }
      else{
	int fiberCh = rs->getInt(19);
	int fiber = rs->getInt(18);
	fCondObject = new HcalElectronicsId(fiberCh, fiber, spigot, dcc);
	fCondObject->setHTR(crate,slot,top);
      }
      if (_obj_name.find("HcalTrigTowerDetId")!=std::string::npos){
	fObject->mapEId2tId(*fCondObject, id);
      }
      else if (_obj_name.find("HcalDetId")!=std::string::npos ||
	       _obj_name.find("HcalCalibDetId")!=std::string::npos ||
	       _obj_name.find("HcalZDCDetId")!=std::string::npos){
	fObject->mapEId2chId(*fCondObject, id);
      }
      else {
	edm::LogWarning("Format Error") << "HcalElectronicsMap-> Unknown subdetector: " 
					<< _obj_name << std::endl; 
      }
      delete fCondObject;
    }
    //Always terminate statement
    connection->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw cms::Exception("ReadError") << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
  }
  fObject->sort ();
  return result;
}


bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalChannelQuality* fObject) {
  std::cout << " +++++=====> HcalDbOmds::getObject" << std::endl;

  bool result=true;
  if (!fObject) return false; //fObject = new HcalChannelQuality;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=12; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel_on_off_state
    // 12. channel_status_word
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      int value = rs->getInt(12);
      //int ieta = rs->getInt(2);
      //int iphi = rs->getInt(3);
      //int depth = rs->getInt(4);
      //HcalSubdetector subdetector = get_subdetector(rs->getString(5));
      //HcalDetId id(subdetector,ieta,iphi,depth);
      //std::cout << "DEBUG: " << std::endl;//<< id << " " << zs << std::endl;
      HcalChannelStatus * fCondObject = new HcalChannelStatus(id, value);
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
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalRespCorrs* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalRespCorrs;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=12; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12. value
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      float value = rs->getFloat(12);
      //int ieta = rs->getInt(2);
      //int iphi = rs->getInt(3);
      //int depth = rs->getInt(4);
      //HcalSubdetector subdetector = get_subdetector(rs->getString(5));
      //HcalDetId id(subdetector,ieta,iphi,depth);
      //std::cout << "DEBUG: " << id << " " << value << std::endl;
      HcalRespCorr * fCondObject = new HcalRespCorr(id, value);
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

// Oracle database connection ownership is transferred here, DO terminate after use
bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalZSThresholds* fObject) {
  bool result=true;
  if (!fObject) return false;// fObject = new HcalZSThresholds;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=12; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12. zs_threshold
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      int zs = rs->getInt(12);
      //int ieta = rs->getInt(2);
      //int iphi = rs->getInt(3);
      //int depth = rs->getInt(4);
      //HcalSubdetector subdetector = get_subdetector(rs->getString(5));
      //HcalDetId id(subdetector,ieta,iphi,depth);
      //std::cout << "DEBUG: " << id << " " << zs << std::endl;
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
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalL1TriggerObjects* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalL1TriggerObjects;
  std::string _tag;
  std::string _algo;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    rs->setMaxColumnSize(1,128);
    rs->setMaxColumnSize(2,128);
    rs->setMaxColumnSize(3,128);
    rs->setMaxColumnSize(4,128);
    rs->setMaxColumnSize(5,128);
    rs->setMaxColumnSize(6,128);
    rs->setMaxColumnSize(7,128);
    rs->setMaxColumnSize(8,128);
    rs->setMaxColumnSize(9,128);
    rs->setMaxColumnSize(10,128);
    rs->setMaxColumnSize(11,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // This is two queries in one joined by "UNION" (SQL), not ordered
    // One of the queries returns global data: LUT tag name and algo name
    // The global data is one raw, and it is ordered in the following way
    // 1."fakeobjectname",
    // 2."fakesubdetector", 3. -1, 4. -1, 5. -1, 6. -1, 7."fakesection", 8. -1, 9. -1, 10. -1, 11. -1 
    // 12. -999999.0, 13. -999999.0, 14. -999999,
    // 15. TRIGGER_OBJECT_METADATA_NAME, 16. TRIGGER_OBJECT_METADATA_VALUE
    //
    // The channel query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12. AVERAGE_PEDESTAL, 13. RESPONSE_CORRECTED_GAIN, 14. FLAG,
    // 15. 'fake_metadata_name', 16. 'fake_metadata_value'
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      float average_pedestal = rs->getFloat(12);
      float response_corrected_gain = rs->getFloat(13);
      int flag = rs->getInt(14);
      std::string metadata_name = rs->getString(15);
      if (metadata_name.find("lut_tag")!=std::string::npos){
	_tag = rs->getString(16);
      }
      else if (metadata_name.find("algo_name")!=std::string::npos){
	_algo = rs->getString(16);
      }
      HcalL1TriggerObject * fCondObject = new HcalL1TriggerObject(id, average_pedestal, response_corrected_gain, flag);
      fObject->addValues(*fCondObject);
      delete fCondObject;
    }
    //Always terminate statement
    connection->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw cms::Exception("ReadError") << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
  }
  fObject->setTagString(_tag);
  fObject->setAlgoString(_algo);
  return result;
}

bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalValidationCorrs* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalValidationCorrs;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    //stmt->setString(2,fVersion);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=12; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12. value
    //
    while (rs->next()) {
      _row.count();
      DetId id = getId(rs);
      float value = rs->getFloat(12);
      //int ieta = rs->getInt(2);
      //int iphi = rs->getInt(3);
      //int depth = rs->getInt(4);
      //HcalSubdetector subdetector = get_subdetector(rs->getString(5));
      //HcalDetId id(subdetector,ieta,iphi,depth);
      //std::cout << "DEBUG: " << id << " " << value << std::endl;
      HcalValidationCorr * fCondObject = new HcalValidationCorr(id, value);
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
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalLutMetadata* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalLutMetadata;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    //stmt->setString(2,fVersion);
    stmt->setInt(2,fIOVBegin);

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=12; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL channels processed: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // This is two queries in one joined by "UNION" (SQL), not ordered
    // One of the queries returns global data: RCTLSB and nominal gain.
    // The global data is one raw, and it is ordered in the following way
    // 1."fakeobjectname",
    // 2."fakesubdetector", 3. -1, 4. -1, 5. -1, 6. -1, 7."fakesection", 8. -1, 9. -1, 10. -1, 11. -1 
    // 12. RCTLSB, 13. nominal_gain, 14. -1
    //
    // The channel query result must be ordered in the following way
    // 1.objectname, ( values: HcalDetId, HcalCalibDetId, HcalTrigTowerDetId, HcalZDCDetId or HcalCastorDetId)
    // 2.subdet, 3.ieta, 4.iphi, 5.depth, 6.type, 7.section, 8.ispositiveeta, 9.sector, 10.module, 11.channel 
    // 12. rec_hit_calibration, 13. lut_granularity, 14. output_lut_threshold 
    //
    while (rs->next()) {
      if (rs->getString(1).find("fakeobjectname")!=std::string::npos){ // global data
	float rctlsb = rs->getFloat(12);
	float nominal_gain = rs->getFloat(13);
	fObject->setRctLsb(rctlsb);
	fObject->setNominalGain(nominal_gain);
      }
      else{ // channel data
	_row.count();
	DetId id = getId(rs);
	float rcalib = rs->getFloat(12);
	uint32_t lut_granularity = rs->getInt(13);
	uint32_t output_lut_threshold = rs->getInt(14);
	HcalLutMetadatum * fCondObject = new HcalLutMetadatum(id,
							      rcalib,
							      lut_granularity,
							      output_lut_threshold);
	fObject->addValues(*fCondObject);
	delete fCondObject;
      }
    }
    //Always terminate statement
    connection->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw cms::Exception("ReadError") << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
  }
  return result;
}

// version is needed for the DCS (unlike most other conditions)
bool HcalDbOmds::getObject (oracle::occi::Connection * connection, 
			    const std::string & fTag, 
			    const std::string & fVersion,
			    const int fSubversion,
			    const int fIOVBegin,
			    const std::string & fQuery,
			    HcalDcsValues* fObject) {
  bool result=true;
  if (!fObject) return false; //fObject = new HcalDcsValues;
  try {
    oracle::occi::Statement* stmt = connection->createStatement(fQuery);
    stmt->setString(1,fTag);
    stmt->setString(2,fVersion);
    stmt->setInt(3,fSubversion);
    stmt->setInt(4,fIOVBegin);

    std::cout << "DEBUG****** IOV=" << fIOVBegin << std::endl;

    ResultSet *rs = stmt->executeQuery();

    // protection agains NULL values
    for (int _i=1; _i!=9; _i++) rs->setMaxColumnSize(_i,128);

    RooGKCounter _row(1,100);
    _row.setMessage("HCAL DCS records: ");
    _row.setPrintCount(true);
    _row.setNewLine(true);
    //
    // The query for the DCS summary condition
    //
    // The channel query result must be ordered in the following way
    //
    // 1. dpname (string)
    // 2. lumi section,
    // 3. value,
    // 4. upper limit,
    // 5. lower limit,
    // 6. subdetector (string)
    // 7. side_ring
    // 8. slice
    // 9. subchannel
    // 10. type (string)
    //
    while (rs->next()) {
      _row.count();

      std::string _dpname = rs->getString(1);
      //HcalOtherSubdetector subd      = getSubDetFromDpName(_dpname);
      //int sidering                   = getSideRingFromDpName(_dpname);
      //unsigned int slice             = getSliceFromDpName(_dpname);
      //HcalDcsDetId::DcsType dcs_type = getDcsTypeFromDpName(_dpname);
      //unsigned int subchan           = getSubChannelFromDpName(_dpname);

      HcalOtherSubdetector subd      = getSubDetFromString(rs->getString(6));
      int sidering                   = rs->getInt(7);
      unsigned int slice             = rs->getInt(8);
      unsigned int subchan           = rs->getInt(9);
      HcalDcsDetId::DcsType dcs_type = getDcsTypeFromString(rs->getString(10));

      HcalDcsDetId newId(subd, sidering, slice, 
			 dcs_type, subchan);

      int LS = rs->getInt(2);
      float val = rs->getFloat(3);
      float upper = rs->getFloat(4);
      float lower = rs->getFloat(5);

      HcalDcsValue * fCondObject = new HcalDcsValue(newId.rawId(),
						    LS,
						    val,
						    upper,
						    lower);

      if (!(fObject->addValue(*fCondObject))) {
	edm::LogWarning("Data Error") << "Data for data point " << _dpname
				      << "\nwas not added to the HcalDcsValues object." << std::endl;
      }
      delete fCondObject;
    }
    fObject->sortAll();
    //Always terminate statement
    connection->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw cms::Exception("ReadError") << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
  }
  return result;
}


bool HcalDbOmds::dumpObject (std::ostream& fOutput, const HcalZSThresholds& fObject) {
  return true;
}


HcalSubdetector HcalDbOmds::get_subdetector( std::string _det )
{
  HcalSubdetector result;
  if      ( _det.find("HB") != std::string::npos ) result = HcalBarrel;
  else if ( _det.find("HE") != std::string::npos ) result = HcalEndcap;
  else if ( _det.find("HF") != std::string::npos ) result = HcalForward;
  else if ( _det.find("HO") != std::string::npos ) result = HcalOuter;
  else                                        result = HcalOther;  

  return result;
}

HcalZDCDetId::Section HcalDbOmds::get_zdc_section( std::string _section )
{
  HcalZDCDetId::Section result;
  if      ( _section.find("Unknown") != std::string::npos ) result = HcalZDCDetId::Unknown;
  else if ( _section.find("EM") != std::string::npos )   result = HcalZDCDetId::EM;
  else if ( _section.find("HAD") != std::string::npos )  result = HcalZDCDetId::HAD;
  else if ( _section.find("LUM") != std::string::npos ) result = HcalZDCDetId::LUM;
  else                                              result = HcalZDCDetId::Unknown;  
  
  return result;
}


HcalOtherSubdetector HcalDbOmds::getSubDetFromDpName(std::string _dpname){
  HcalOtherSubdetector subd;
  switch (_dpname.at(_dpname.find("HVcrate_")+9)) {
  case 'B':
    subd = HcalDcsBarrel;
    break;
  case 'E':
    subd = HcalDcsEndcap;
    break;
  case 'F':
    subd = HcalDcsForward;
    break;
  case 'O':
    subd = HcalDcsOuter;
    break;
  default:
    subd = HcalOtherEmpty;
    break;
  }
  return subd;
}

int HcalDbOmds::getSideRingFromDpName(std::string _dpname){
  int _side_ring = 1000;
  int _side = 10;
  int _ring = 100;
  switch (_dpname.at(_dpname.find("HVcrate_")+9)) {
  case 'B':
  case 'E':
  case 'F':
    if (_dpname.at(_dpname.find("HVcrate_")+10) == 'M') _side = -1;
    else if (_dpname.at(_dpname.find("HVcrate_")+10) == 'P') _side = +1;
    _ring = 1;
      break;
  case 'O':
    if (_dpname.at(_dpname.find("HVcrate_")+11) == 'M') _side = -1;
    else if (_dpname.at(_dpname.find("HVcrate_")+11) == 'P') _side = +1;
    _ring = atoi(&(_dpname.at(_dpname.find("HVcrate_")+10)));
    break;
  default:
    break;
  }
  _side_ring = _side * _ring;
  return _side_ring;
}

unsigned int HcalDbOmds::getSliceFromDpName(std::string _dpname){
  int result = 1000;
  HcalDbOmds::from_string<int>(result, _dpname.substr(_dpname.find("/S")+2,2), std::dec);
  return (unsigned int)result;
}

unsigned int HcalDbOmds::getSubChannelFromDpName(std::string _dpname){
  unsigned int result = 1000;
  HcalDbOmds::from_string<unsigned int>(result, _dpname.substr(_dpname.find("/RM")+3,1), std::dec);
  return result;
}

// FIXME: adjust to new PVSS data point naming convention
// together with DcsDetId
HcalDcsDetId::DcsType HcalDbOmds::getDcsTypeFromDpName(std::string _dpname){
  HcalDcsDetId::DcsType result = HcalDcsDetId::DcsType(15); // unknown
  std::string _type = _dpname.substr(_dpname.find("/RM")+4);
  if (_type.find("HV")!=std::string::npos) result = HcalDcsDetId::DcsType(1);
  return result;
}

HcalOtherSubdetector HcalDbOmds::getSubDetFromString(std::string subdet){
  HcalOtherSubdetector subd;
  switch (subdet.at(1)){
  case 'B':
    subd = HcalDcsBarrel;
    break;
  case 'E':
    subd = HcalDcsEndcap;
    break;
  case 'F':
    subd = HcalDcsForward;
    break;
  case 'O':
    subd = HcalDcsOuter;
    break;
  default:
    subd = HcalOtherEmpty;
    break;
  }
  return subd;
}

HcalDcsDetId::DcsType HcalDbOmds::getDcsTypeFromString(std::string type){
  HcalDcsDetId::DcsType result = HcalDcsDetId::DCSUNKNOWN; // unknown
  if (type.find("HV")!=std::string::npos) result = HcalDcsDetId::HV;
  else if (type.find("BV")!=std::string::npos) result = HcalDcsDetId::BV;
  else if (type.find("Cath")!=std::string::npos) result = HcalDcsDetId::CATH;
  else if (type.find("Dyn7")!=std::string::npos) result = HcalDcsDetId::DYN7;
  else if (type.find("Dyn8")!=std::string::npos) result = HcalDcsDetId::DYN8;
  else if (type.find("RM_TEMP")!=std::string::npos) result = HcalDcsDetId::RM_TEMP;
  else if (type.find("CCM_TEMP")!=std::string::npos) result = HcalDcsDetId::CCM_TEMP;
  else if (type.find("CALIB_TEMP")!=std::string::npos) result = HcalDcsDetId::CALIB_TEMP;
  else if (type.find("LVTTM_TEMP")!=std::string::npos) result = HcalDcsDetId::LVTTM_TEMP;
  else if (type.find("TEMP")!=std::string::npos) result = HcalDcsDetId::TEMP;
  else if (type.find("QPLL_LOCK")!=std::string::npos) result = HcalDcsDetId::QPLL_LOCK;
  else if (type.find("STATUS")!=std::string::npos) result = HcalDcsDetId::STATUS;
  else if (type.find("DCS_MAX")!=std::string::npos) result = HcalDcsDetId::DCS_MAX;
  return result;
}

