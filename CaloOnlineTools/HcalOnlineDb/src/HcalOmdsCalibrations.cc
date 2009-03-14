// -*- C++ -*-
//
// Original Author:  Gena Kukartsev Mar 11, 2009
// Adapted from HcalOmdsCalibrations
// $Id: HcalOmdsCalibrations.cc,v 1.1 2009/03/13 11:27:37 kukartse Exp $
//
//

#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalDbOmds.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalZSThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/HcalL1TriggerObjectsRcd.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalOmdsCalibrations.h"
//
// class decleration
//

using namespace cms;

HcalOmdsCalibrations::HcalOmdsCalibrations ( const edm::ParameterSet& iConfig ) 
  
{
  //parsing parameters
  std::vector<edm::ParameterSet> data = iConfig.getParameter<std::vector<edm::ParameterSet> >("input");
  std::vector<edm::ParameterSet>::iterator request = data.begin ();
  for (; request != data.end (); request++) {
    std::string objectName = request->getParameter<std::string> ("object");
    //edm::FileInPath fp = request->getParameter<edm::FileInPath>("file");
    //mInputs [objectName] = fp.fullPath();
    mInputs [objectName] = request->getParameter<std::string>("tag");
    if (objectName == "Pedestals") {
      setWhatProduced (this, &HcalOmdsCalibrations::producePedestals);
      findingRecord <HcalPedestalsRcd> ();
    }
    else if (objectName == "PedestalWidths") {
      setWhatProduced (this, &HcalOmdsCalibrations::producePedestalWidths);
      findingRecord <HcalPedestalWidthsRcd> ();
    }
    else if (objectName == "Gains") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceGains);
      findingRecord <HcalGainsRcd> ();
    }
    else if (objectName == "GainWidths") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceGainWidths);
      findingRecord <HcalGainWidthsRcd> ();
    }
    else if (objectName == "QIEData") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceQIEData);
      findingRecord <HcalQIEDataRcd> ();
    }
    else if (objectName == "ChannelQuality") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceChannelQuality);
      findingRecord <HcalChannelQualityRcd> ();
    }
    else if (objectName == "ZSThresholds") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceZSThresholds);
      findingRecord <HcalZSThresholdsRcd> ();
    }
    else if (objectName == "RespCorrs") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceRespCorrs);
      findingRecord <HcalRespCorrsRcd> ();
    }
    else if (objectName == "L1TriggerObjects") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceL1TriggerObjects);
      findingRecord <HcalL1TriggerObjectsRcd> ();
    }
    else if (objectName == "ElectronicsMap") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceElectronicsMap);
      findingRecord <HcalElectronicsMapRcd> ();
    }
    else {
      std::cerr << "HcalOmdsCalibrations-> Unknown object name '" << objectName 
		<< "', known names are: "
		<< "Pedestals PedestalWidths Gains GainWidths QIEData ChannelQuality ElectronicsMap "
		<< "ZSThresholds RespCorrs L1TriggerObjects"
		<< std::endl;
    }
  }
  //  setWhatProduced(this);
}


HcalOmdsCalibrations::~HcalOmdsCalibrations()
{
}


//
// member functions
//
void 
HcalOmdsCalibrations::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  std::string record = iKey.name ();
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

// FIXME: put this into the HcalDbOmds namespace
const static std::string omds_occi_default_accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
//const static std::string omds_occi_default_accessor = "occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44,LHWM_VERSION=22";

template <class T>
std::auto_ptr<T> produce_impl (const std::string & fTag, const std::string& fAccessor = omds_occi_default_accessor) {
  std::auto_ptr<T> result (new T ());

  HCALConfigDB * db = new HCALConfigDB();
  try {
    // FIXME: check that all disconnects, releases and cleanup is done in the end
    db -> connect( fAccessor );
  } catch (hcal::exception::ConfigurationDatabaseException & e) {
    std::cerr << "Cannot connect to the database" << endl;
  }
  oracle::occi::Connection * _connection = db -> getConnection();
  if (_connection){
    if (!HcalDbOmds::getObject (_connection, fTag, &*result)) {
      std::cerr << "HcalOmdsCalibrations-> Can not read tag name '" << fTag << "' from database '" << fAccessor << "'" << std::endl;
      throw cms::Exception("ReadError") << "Can not read tag name '" << fTag << "' from database '" << fAccessor << "'" << std::endl;
    }
  }
  else{
    std::cerr << "Database connection is null. This should NEVER happen here. Something fishy is going on..." << std::endl;
  }
  return result;
}



std::auto_ptr<HcalPedestals> HcalOmdsCalibrations::producePedestals (const HcalPedestalsRcd&) {
  return produce_impl<HcalPedestals> (mInputs ["Pedestals"]);
}

std::auto_ptr<HcalPedestalWidths> HcalOmdsCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd&) {
  return produce_impl<HcalPedestalWidths> (mInputs ["PedestalWidths"]);
}

std::auto_ptr<HcalGains> HcalOmdsCalibrations::produceGains (const HcalGainsRcd&) {
  return produce_impl<HcalGains> (mInputs ["Gains"]);
}

std::auto_ptr<HcalGainWidths> HcalOmdsCalibrations::produceGainWidths (const HcalGainWidthsRcd&) {
  return produce_impl<HcalGainWidths> (mInputs ["GainWidths"]);
}

std::auto_ptr<HcalQIEData> HcalOmdsCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  return produce_impl<HcalQIEData> (mInputs ["QIEData"]);
}

std::auto_ptr<HcalChannelQuality> HcalOmdsCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  return produce_impl<HcalChannelQuality> (mInputs ["ChannelQuality"]);
}

std::auto_ptr<HcalZSThresholds> HcalOmdsCalibrations::produceZSThresholds (const HcalZSThresholdsRcd& rcd) {
  return produce_impl<HcalZSThresholds> (mInputs ["ZSThresholds"]);
}

std::auto_ptr<HcalRespCorrs> HcalOmdsCalibrations::produceRespCorrs (const HcalRespCorrsRcd& rcd) {
  return produce_impl<HcalRespCorrs> (mInputs ["RespCorrs"]);
}

std::auto_ptr<HcalL1TriggerObjects> HcalOmdsCalibrations::produceL1TriggerObjects (const HcalL1TriggerObjectsRcd& rcd) {
  return produce_impl<HcalL1TriggerObjects> (mInputs ["L1TriggerObjects"]);
}

std::auto_ptr<HcalElectronicsMap> HcalOmdsCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  return produce_impl<HcalElectronicsMap> (mInputs ["ElectronicsMap"]);
}

