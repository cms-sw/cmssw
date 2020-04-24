// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// Adapted for CASTOR by L. Mundim
//

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/ValidityInterval.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbHardcode.h"

#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"

#include "Geometry/ForwardGeometry/interface/CastorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CastorHardcodeCalibrations.h"

// class declaration
//

using namespace cms;

namespace {

std::vector<HcalGenericDetId> allCells (bool h2_mode) {
  static std::vector<HcalGenericDetId> result;
  if (result.empty()) {

    CastorTopology castortopology;
    HcalCastorDetId cell;
    HcalCastorDetId::Section section  = HcalCastorDetId::EM;

    for(int sector=1; sector<17; sector++) {
      for(int module=1; module<3; module++) {
	cell = HcalCastorDetId(section, true, sector, module);
    if (castortopology.valid(cell)) result.push_back(cell);
    cell = HcalCastorDetId(section, false, sector, module);
    if (castortopology.valid(cell)) result.push_back(cell);
    }
   }

   section = HcalCastorDetId::HAD;
    for(int sector= 1; sector < 17; sector++){
     for(int module=3; module<15; module++) {
      cell = HcalCastorDetId(section, true, sector, module);
      if(castortopology.valid(cell)) result.push_back(cell);
      cell = HcalCastorDetId(section, false, sector, module);
      if(castortopology.valid(cell)) result.push_back(cell);
     }
   }

}
  return result;

 }
}

CastorHardcodeCalibrations::CastorHardcodeCalibrations ( const edm::ParameterSet& iConfig ) 
  
{
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::CastorHardcodeCalibrations->...";
  //parsing record parameters
  h2mode_=iConfig.getUntrackedParameter<bool>("H2Mode",false);
  std::vector <std::string> toGet = iConfig.getUntrackedParameter <std::vector <std::string> > ("toGet");
  for(std::vector <std::string>::iterator objectName = toGet.begin(); objectName != toGet.end(); ++objectName ) {
    bool all = *objectName == "all";
    if ((*objectName == "Pedestals") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::producePedestals);
      findingRecord <CastorPedestalsRcd> ();
    }
    if ((*objectName == "PedestalWidths") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::producePedestalWidths);
      findingRecord <CastorPedestalWidthsRcd> ();
    }
    if ((*objectName == "Gains") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::produceGains);
      findingRecord <CastorGainsRcd> ();
    }
    if ((*objectName == "GainWidths") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::produceGainWidths);
      findingRecord <CastorGainWidthsRcd> ();
    }
    if ((*objectName == "QIEData") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::produceQIEData);
      findingRecord <CastorQIEDataRcd> ();
    }
    if ((*objectName == "ChannelQuality") || (*objectName == "channelQuality") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::produceChannelQuality);
      findingRecord <CastorChannelQualityRcd> ();
    }
    if ((*objectName == "ElectronicsMap") || (*objectName == "electronicsMap") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::produceElectronicsMap);
      findingRecord <CastorElectronicsMapRcd> ();
    }
    if ((*objectName == "RecoParams") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::produceRecoParams);
      findingRecord <CastorRecoParamsRcd> ();
    }
    if ((*objectName == "SaturationCorrs") || all) {
      setWhatProduced (this, &CastorHardcodeCalibrations::produceSaturationCorrs);
      findingRecord <CastorSaturationCorrsRcd> ();
    }
  }
}


CastorHardcodeCalibrations::~CastorHardcodeCalibrations()
{
}


//
// member functions
//
void 
CastorHardcodeCalibrations::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  std::string record = iKey.name ();
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::setIntervalFor-> key: " << record << " time: " << iTime.eventID() << '/' << iTime.time ().value ();
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

std::unique_ptr<CastorPedestals> CastorHardcodeCalibrations::producePedestals (const CastorPedestalsRcd&) {
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::producePedestals-> ...";
  auto result = std::make_unique<CastorPedestals>(false);
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    CastorPedestal item = CastorDbHardcode::makePedestal (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<CastorPedestalWidths> CastorHardcodeCalibrations::producePedestalWidths (const CastorPedestalWidthsRcd&) {
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::producePedestalWidths-> ...";
  auto result = std::make_unique<CastorPedestalWidths>(false);
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    CastorPedestalWidth item = CastorDbHardcode::makePedestalWidth (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<CastorGains> CastorHardcodeCalibrations::produceGains (const CastorGainsRcd&) {
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::produceGains-> ...";
  auto result = std::make_unique<CastorGains>();
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    CastorGain item = CastorDbHardcode::makeGain (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<CastorGainWidths> CastorHardcodeCalibrations::produceGainWidths (const CastorGainWidthsRcd&) {
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::produceGainWidths-> ...";
  auto result = std::make_unique<CastorGainWidths>();
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    CastorGainWidth item = CastorDbHardcode::makeGainWidth (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<CastorQIEData> CastorHardcodeCalibrations::produceQIEData (const CastorQIEDataRcd& rcd) {
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::produceQIEData-> ...";
  auto result = std::make_unique<CastorQIEData>();
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    CastorQIECoder coder = CastorDbHardcode::makeQIECoder (*cell);
    result->addCoder(coder);
  }
  return result;
}

std::unique_ptr<CastorChannelQuality> CastorHardcodeCalibrations::produceChannelQuality (const CastorChannelQualityRcd& rcd) {
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::produceChannelQuality-> ...";
  auto result = std::make_unique<CastorChannelQuality>();
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    CastorChannelStatus item(cell->rawId(),CastorChannelStatus::GOOD);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<CastorElectronicsMap> CastorHardcodeCalibrations::produceElectronicsMap (const CastorElectronicsMapRcd& rcd) {
  edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::produceElectronicsMap-> ...";

  auto result = std::make_unique<CastorElectronicsMap>();
  CastorDbHardcode::makeHardcodeMap(*result);
  return result;
}

std::unique_ptr<CastorRecoParams> CastorHardcodeCalibrations::produceRecoParams (const CastorRecoParamsRcd& rcd) {
	edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::produceRecoParams-> ...";
        auto result = std::make_unique<CastorRecoParams>();
	std::vector <HcalGenericDetId> cells = allCells(h2mode_);
	for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
		CastorRecoParam item = CastorDbHardcode::makeRecoParam (*cell);
		result->addValues(item);
	}
	return result;
}

std::unique_ptr<CastorSaturationCorrs> CastorHardcodeCalibrations::produceSaturationCorrs (const CastorSaturationCorrsRcd& rcd) {
	edm::LogInfo("HCAL") << "CastorHardcodeCalibrations::produceSaturationCorrs-> ...";
        auto result = std::make_unique<CastorSaturationCorrs>();
	std::vector <HcalGenericDetId> cells = allCells(h2mode_);
	for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
		CastorSaturationCorr item = CastorDbHardcode::makeSaturationCorr (*cell);
		result->addValues(item);
	}
	return result;
}
