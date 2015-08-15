// -*- C++ -*-
//
// Original Author:  Gena Kukartsev Mar 11, 2009
// Adapted from HcalAsciiCalibrations
//

#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/ESHandle.h"

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
#include "CondFormats/DataRecord/interface/HcalValidationCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"
#include "CondFormats/DataRecord/interface/HcalDcsRcd.h"

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

    if (request->exists("version")) mVersion[objectName] = request->getParameter<std::string>("version");
    if (request->exists("subversion")) mSubversion[objectName] = request->getParameter<int>("subversion");
    if (request->exists("iov_begin")) mIOVBegin[objectName] = request->getParameter<int>("iov_begin");
    if (request->exists("accessor")) mAccessor[objectName] = request->getParameter<std::string>("accessor");
    if (request->exists("query")) mQuery[objectName] = request->getParameter<std::string>("query");

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
    else if (objectName == "ValidationCorrs") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceValidationCorrs);
      findingRecord <HcalValidationCorrsRcd> ();
    }
    else if (objectName == "LutMetadata") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceLutMetadata);
      findingRecord <HcalLutMetadataRcd> ();
    }
    else if (objectName == "DcsValues") {
      setWhatProduced (this, &HcalOmdsCalibrations::produceDcsValues);
      findingRecord <HcalDcsRcd> ();
    }
    else {
      std::cerr << "HcalOmdsCalibrations-> Unknown object name '" << objectName 
		<< "', known names are: "
		<< "Pedestals PedestalWidths Gains GainWidths QIEData ChannelQuality ElectronicsMap "
		<< "ZSThresholds RespCorrs L1TriggerObjects ValidationCorrs LutMetadata DcsValues "
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
//const static std::string omds_occi_default_accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
const static std::string omds_occi_default_accessor = "occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44,LHWM_VERSION=22";
const static std::string default_version = "";
const static std::string default_query = "";

template <class T>
std::auto_ptr<T> produce_impl (const HcalTopology* topo,
			       const std::string & fTag, 
			       const std::string & fVersion=default_version, 
			       const int fSubversion=1, 
			       const int fIOVBegin=1,
			       const std::string & fQuery = default_query, 
			       const std::string& fAccessor = omds_occi_default_accessor ) {
  std::auto_ptr<T> result (new T (topo));

  HCALConfigDB * db = new HCALConfigDB();
  try {
    db -> connect( fAccessor );
  } catch (hcal::exception::ConfigurationDatabaseException & e) {
    std::cerr << "Cannot connect to the database" << std::endl;
  }
  oracle::occi::Connection * _connection = db -> getConnection();
  if (_connection){
    if (!HcalDbOmds::getObject (_connection, fTag, fVersion, fSubversion, fIOVBegin, fQuery, &*result)) {
      std::cerr << "HcalOmdsCalibrations-> Can not read tag name '" << fTag << "' from database '" << fAccessor << "'" << std::endl;
      throw cms::Exception("ReadError") << "Can not read tag name '" << fTag << "' from database '" << fAccessor << "'" << std::endl;
    }
  }
  else{
    std::cerr << "Database connection is null. This should NEVER happen here. Something fishy is going on..." << std::endl;
  }
  return result;
}
template <class T>
std::auto_ptr<T> produce_impl (const std::string & fTag, 
			       const std::string & fVersion=default_version, 
			       const int fSubversion=1, 
			       const int fIOVBegin=1,
			       const std::string & fQuery = default_query, 
			       const std::string& fAccessor = omds_occi_default_accessor ) {
  std::auto_ptr<T> result (new T ());

  HCALConfigDB * db = new HCALConfigDB();
  try {
    db -> connect( fAccessor );
  } catch (hcal::exception::ConfigurationDatabaseException & e) {
    std::cerr << "Cannot connect to the database" << std::endl;
  }
  oracle::occi::Connection * _connection = db -> getConnection();
  if (_connection){
    if (!HcalDbOmds::getObject (_connection, fTag, fVersion, fSubversion, fIOVBegin, fQuery, &*result)) {
      std::cerr << "HcalOmdsCalibrations-> Can not read tag name '" << fTag << "' from database '" << fAccessor << "'" << std::endl;
      throw cms::Exception("ReadError") << "Can not read tag name '" << fTag << "' from database '" << fAccessor << "'" << std::endl;
    }
  }
  else{
    std::cerr << "Database connection is null. This should NEVER happen here. Something fishy is going on..." << std::endl;
  }
  return result;
}



std::auto_ptr<HcalPedestals> HcalOmdsCalibrations::producePedestals (const HcalPedestalsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  return produce_impl<HcalPedestals> (topo, mInputs ["Pedestals"],
				      mVersion["Pedestals"], 
				      mSubversion["Pedestals"], 
				      mIOVBegin["Pedestals"], 
				      mQuery["Pedestals"], 
				      mAccessor["Pedestals"]);
}

std::auto_ptr<HcalPedestalWidths> HcalOmdsCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalPedestalWidths> (topo, mInputs ["PedestalWidths"], 
					   mVersion["PedestalWidths"], 
					   mSubversion["PedestalWidths"], 
					   mIOVBegin["PedestalWidths"], 
					   mQuery["PedestalWidths"], 
					   mAccessor["PedestalWidths"]);
}

std::auto_ptr<HcalGains> HcalOmdsCalibrations::produceGains (const HcalGainsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalGains> (topo, mInputs ["Gains"], 
				  mVersion["Gains"], 
				  mSubversion["Gains"], 
				  mIOVBegin["Gains"], 
				  mQuery["Gains"], 
				  mAccessor["Gains"]);
}

std::auto_ptr<HcalGainWidths> HcalOmdsCalibrations::produceGainWidths (const HcalGainWidthsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalGainWidths> (topo, mInputs ["GainWidths"], 
				       mVersion["GainWidths"], 
				       mSubversion["GainWidths"], 
				       mIOVBegin["GainWidths"], 
				       mQuery["GainWidths"], 
				       mAccessor["GainWidths"]);
}

std::auto_ptr<HcalQIEData> HcalOmdsCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalQIEData> (topo, mInputs ["QIEData"], 
				    mVersion["QIEData"], 
				    mSubversion["QIEData"], 
				    mIOVBegin["QIEData"], 
				    mQuery["QIEData"], 
				    mAccessor["QIEData"]);
}

std::auto_ptr<HcalChannelQuality> HcalOmdsCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalChannelQuality> (topo, mInputs ["ChannelQuality"], 
					   mVersion["ChannelQuality"], 
					   mSubversion["ChannelQuality"], 
					   mIOVBegin["ChannelQuality"], 
					   mQuery["ChannelQuality"], 
					   mAccessor["ChannelQuality"]);
}

std::auto_ptr<HcalZSThresholds> HcalOmdsCalibrations::produceZSThresholds (const HcalZSThresholdsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalZSThresholds> (topo, mInputs ["ZSThresholds"], 
					 mVersion["ZSThresholds"], 
					 mSubversion["ZSThresholds"], 
					 mIOVBegin["ZSThresholds"], 
					 mQuery["ZSThresholds"], 
					 mAccessor["ZSThresholds"]);
}

std::auto_ptr<HcalRespCorrs> HcalOmdsCalibrations::produceRespCorrs (const HcalRespCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalRespCorrs> (topo, mInputs ["RespCorrs"], 
				      mVersion["RespCorrs"], 
				      mSubversion["RespCorrs"], 
				      mIOVBegin["RespCorrs"], 
				      mQuery["RespCorrs"], 
				      mAccessor["RespCorrs"]);
}

std::auto_ptr<HcalL1TriggerObjects> HcalOmdsCalibrations::produceL1TriggerObjects (const HcalL1TriggerObjectsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalL1TriggerObjects> (topo, mInputs ["L1TriggerObjects"], 
					     mVersion["L1TriggerObjects"], 
					     mSubversion["L1TriggerObjects"], 
					     mIOVBegin["L1TriggerObjects"], 
					     mQuery["L1TriggerObjects"], 
					     mAccessor["L1TriggerObjects"]);
}

std::auto_ptr<HcalElectronicsMap> HcalOmdsCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  return produce_impl<HcalElectronicsMap> (mInputs ["ElectronicsMap"], 
					   mVersion["ElectronicsMap"], 
					   mSubversion["ElectronicsMap"], 
					   mIOVBegin["ElectronicsMap"], 
					   mQuery["ElectronicsMap"], 
					   mAccessor["ElectronicsMap"]);
}

std::auto_ptr<HcalValidationCorrs> HcalOmdsCalibrations::produceValidationCorrs (const HcalValidationCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalValidationCorrs> (topo, mInputs ["ValidationCorrs"], 
					    mVersion["ValidationCorrs"], 
					    mSubversion["ValidationCorrs"], 
					    mIOVBegin["ValidationCorrs"], 
					    mQuery["ValidationCorrs"], 
					    mAccessor["ValidationCorrs"]);
}

std::auto_ptr<HcalLutMetadata> HcalOmdsCalibrations::produceLutMetadata (const HcalLutMetadataRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalLutMetadata> (topo, mInputs ["LutMetadata"], 
					mVersion["LutMetadata"], 
					mSubversion["LutMetadata"], 
					mIOVBegin["LutMetadata"], 
					mQuery["LutMetadata"], 
					mAccessor["LutMetadata"]);
}

std::auto_ptr<HcalDcsValues> HcalOmdsCalibrations::produceDcsValues (const HcalDcsRcd& rcd) {
  return produce_impl<HcalDcsValues> (mInputs ["DcsValues"], 
					mVersion["DcsValues"], 
					mSubversion["DcsValues"], 
					mIOVBegin["DcsValues"], 
					mQuery["DcsValues"], 
					mAccessor["DcsValues"]);
}

