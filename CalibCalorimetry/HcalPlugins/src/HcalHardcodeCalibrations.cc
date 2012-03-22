// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// $Id: HcalHardcodeCalibrations.cc,v 1.28 2011/10/26 14:00:29 xiezhen Exp $
//
//

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/ValidityInterval.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"

#include "CondFormats/DataRecord/interface/HcalAllRcds.h"

#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HcalHardcodeCalibrations.h"

// class decleration
//

using namespace cms;

namespace {

std::vector<HcalGenericDetId> allCells (HcalTopology::Mode mode) {
  static std::vector<HcalGenericDetId> result;
  if (result.size () <= 0) {
    HcalTopology hcaltopology(mode);
    for (int eta = -50; eta < 50; eta++) {
      for (int phi = 0; phi < 100; phi++) {
	for (int depth = 1; depth <= 7; depth++) {
	  for (int det = 1; det < 5; det++) {
	    HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	    if (hcaltopology.valid(cell)) result.push_back (cell);
	  }
	}
      }
    } 
    ZdcTopology zdctopology;
    HcalZDCDetId zcell;
    HcalZDCDetId::Section section  = HcalZDCDetId::EM;
    for(int depth= 1; depth < 6; depth++){
      zcell = HcalZDCDetId(section, true, depth);
      if(zdctopology.valid(zcell)) result.push_back(zcell);
      zcell = HcalZDCDetId(section, false, depth);
      if(zdctopology.valid(zcell)) result.push_back(zcell);     
    }
    section = HcalZDCDetId::HAD;
    for(int depth= 1; depth < 5; depth++){
      zcell = HcalZDCDetId(section, true, depth);
      if(zdctopology.valid(zcell)) result.push_back(zcell);
      zcell = HcalZDCDetId(section, false, depth);
      if(zdctopology.valid(zcell)) result.push_back(zcell);     
    }
    section = HcalZDCDetId::LUM;
    for(int depth= 1; depth < 3; depth++){
      zcell = HcalZDCDetId(section, true, depth);
      if(zdctopology.valid(zcell)) result.push_back(zcell);
      zcell = HcalZDCDetId(section, false, depth);
      if(zdctopology.valid(zcell)) result.push_back(zcell);     
    }
  }
  return result;
}

}

HcalHardcodeCalibrations::HcalHardcodeCalibrations ( const edm::ParameterSet& iConfig ) 
  
{
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::HcalHardcodeCalibrations->...";
  //parsing record parameters
  bool h2mode=iConfig.getUntrackedParameter<bool>("H2Mode",false);
  bool slhcmode=iConfig.getUntrackedParameter<bool>("SLHCMode",false);
  bool h2hemode=iConfig.getUntrackedParameter<bool>("H2HEMode",false);
  if (h2hemode)      mode_=HcalTopology::md_H2HE;
  else if (slhcmode) mode_=HcalTopology::md_SLHC;
  else if (h2mode)   mode_=HcalTopology::md_H2;
  else               mode_=HcalTopology::md_LHC;

  std::vector <std::string> toGet = iConfig.getUntrackedParameter <std::vector <std::string> > ("toGet");
  for(std::vector <std::string>::iterator objectName = toGet.begin(); objectName != toGet.end(); ++objectName ) {
    bool all = *objectName == "all";
    if ((*objectName == "Pedestals") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::producePedestals);
      findingRecord <HcalPedestalsRcd> ();
    }
    if ((*objectName == "PedestalWidths") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::producePedestalWidths);
      findingRecord <HcalPedestalWidthsRcd> ();
    }
    if ((*objectName == "Gains") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceGains);
      findingRecord <HcalGainsRcd> ();
    }
    if ((*objectName == "GainWidths") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceGainWidths);
      findingRecord <HcalGainWidthsRcd> ();
    }
    if ((*objectName == "QIEData") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceQIEData);
      findingRecord <HcalQIEDataRcd> ();
    }
    if ((*objectName == "ChannelQuality") || (*objectName == "channelQuality") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceChannelQuality);
      findingRecord <HcalChannelQualityRcd> ();
    }
    if ((*objectName == "ElectronicsMap") || (*objectName == "electronicsMap") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceElectronicsMap);
      findingRecord <HcalElectronicsMapRcd> ();
    }
    if ((*objectName == "ZSThresholds") || (*objectName == "zsThresholds") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceZSThresholds);
      findingRecord <HcalZSThresholdsRcd> ();
    }
    if ((*objectName == "RespCorrs") || (*objectName == "ResponseCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceRespCorrs);
      findingRecord <HcalRespCorrsRcd> ();
    }
    if ((*objectName == "LUTCorrs") || (*objectName == "LUTCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceLUTCorrs);
      findingRecord <HcalLUTCorrsRcd> ();
    }
    if ((*objectName == "PFCorrs") || (*objectName == "PFCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::producePFCorrs);
      findingRecord <HcalPFCorrsRcd> ();
    }
    if ((*objectName == "TimeCorrs") || (*objectName == "TimeCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceTimeCorrs);
      findingRecord <HcalTimeCorrsRcd> ();
    }
    if ((*objectName == "L1TriggerObjects") || (*objectName == "L1Trigger") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceL1TriggerObjects);
      findingRecord <HcalL1TriggerObjectsRcd> ();
    }
    if ((*objectName == "ValidationCorrs") || (*objectName == "ValidationCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceValidationCorrs);
      findingRecord <HcalValidationCorrsRcd> ();
    }
    if ((*objectName == "LutMetadata") || (*objectName == "lutMetadata") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceLutMetadata);
      findingRecord <HcalLutMetadataRcd> ();
    }
    if ((*objectName == "DcsValues") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceDcsValues);
      findingRecord <HcalDcsRcd> ();
    }
    if ((*objectName == "DcsMap") || (*objectName == "dcsMap") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceDcsMap);
      findingRecord <HcalDcsMapRcd> ();
    }
    if ((*objectName == "RecoParams") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceRecoParams);
      findingRecord <HcalRecoParamsRcd> ();
    }
    if ((*objectName == "LongRecoParams") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceLongRecoParams);
      findingRecord <HcalLongRecoParamsRcd> ();
    }
    if ((*objectName == "MCParams") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceMCParams);
      findingRecord <HcalMCParamsRcd> ();
    }
    if ((*objectName == "FlagHFDigiTimeParams") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceFlagHFDigiTimeParams);
      findingRecord <HcalFlagHFDigiTimeParamsRcd> ();
    }
  }
}


HcalHardcodeCalibrations::~HcalHardcodeCalibrations()
{
}


//
// member functions
//
void 
HcalHardcodeCalibrations::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  std::string record = iKey.name ();
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::setIntervalFor-> key: " << record << " time: " << iTime.eventID() << '/' << iTime.time ().value ();
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

std::auto_ptr<HcalPedestals> HcalHardcodeCalibrations::producePedestals (const HcalPedestalsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePedestals-> ...";
  std::auto_ptr<HcalPedestals> result (new HcalPedestals ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPedestal item = HcalDbHardcode::makePedestal (*cell);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalPedestalWidths> HcalHardcodeCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePedestalWidths-> ...";
  std::auto_ptr<HcalPedestalWidths> result (new HcalPedestalWidths ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPedestalWidth item = HcalDbHardcode::makePedestalWidth (*cell);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalGains> HcalHardcodeCalibrations::produceGains (const HcalGainsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceGains-> ...";
  std::auto_ptr<HcalGains> result (new HcalGains ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalGain item = HcalDbHardcode::makeGain (*cell);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalGainWidths> HcalHardcodeCalibrations::produceGainWidths (const HcalGainWidthsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceGainWidths-> ...";
  std::auto_ptr<HcalGainWidths> result (new HcalGainWidths ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalGainWidth item = HcalDbHardcode::makeGainWidth (*cell);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalQIEData> HcalHardcodeCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceQIEData-> ...";
  std::auto_ptr<HcalQIEData> result (new HcalQIEData ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalQIECoder coder = HcalDbHardcode::makeQIECoder (*cell);
    result->addCoder (coder);
  }
  return result;
}

std::auto_ptr<HcalChannelQuality> HcalHardcodeCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceChannelQuality-> ...";
  std::auto_ptr<HcalChannelQuality> result (new HcalChannelQuality ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalChannelStatus item(cell->rawId(),0);
    result->addValues(item);
  }
  return result;
}


std::auto_ptr<HcalRespCorrs> HcalHardcodeCalibrations::produceRespCorrs (const HcalRespCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceRespCorrs-> ...";
  std::auto_ptr<HcalRespCorrs> result (new HcalRespCorrs ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalRespCorr item(cell->rawId(),1.0);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalLUTCorrs> HcalHardcodeCalibrations::produceLUTCorrs (const HcalLUTCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLUTCorrs-> ...";
  std::auto_ptr<HcalLUTCorrs> result (new HcalLUTCorrs ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalLUTCorr item(cell->rawId(),1.0);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalPFCorrs> HcalHardcodeCalibrations::producePFCorrs (const HcalPFCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePFCorrs-> ...";
  std::auto_ptr<HcalPFCorrs> result (new HcalPFCorrs ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPFCorr item(cell->rawId(),1.0);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalTimeCorrs> HcalHardcodeCalibrations::produceTimeCorrs (const HcalTimeCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTimeCorrs-> ...";
  std::auto_ptr<HcalTimeCorrs> result (new HcalTimeCorrs ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalTimeCorr item(cell->rawId(),0.0);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalZSThresholds> HcalHardcodeCalibrations::produceZSThresholds (const HcalZSThresholdsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceZSThresholds-> ...";
  std::auto_ptr<HcalZSThresholds> result (new HcalZSThresholds ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalZSThreshold item(cell->rawId(),0);
    result->addValues(item);
  }
  return result;
}


std::auto_ptr<HcalL1TriggerObjects> HcalHardcodeCalibrations::produceL1TriggerObjects (const HcalL1TriggerObjectsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceL1TriggerObjects-> ...";
  std::auto_ptr<HcalL1TriggerObjects> result (new HcalL1TriggerObjects ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalL1TriggerObject item(cell->rawId(),0., 1., 0);
    result->addValues(item);
  }
  // add tag and algo values
  result->setTagString("hardcoded");
  result->setAlgoString("hardcoded");
  return result;
}




std::auto_ptr<HcalElectronicsMap> HcalHardcodeCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceElectronicsMap-> ...";

  std::auto_ptr<HcalElectronicsMap> result (new HcalElectronicsMap ());
  HcalDbHardcode::makeHardcodeMap(*result);
  return result;
}

std::auto_ptr<HcalValidationCorrs> HcalHardcodeCalibrations::produceValidationCorrs (const HcalValidationCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceValidationCorrs-> ...";
  std::auto_ptr<HcalValidationCorrs> result (new HcalValidationCorrs ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalValidationCorr item(cell->rawId(),1.0);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalLutMetadata> HcalHardcodeCalibrations::produceLutMetadata (const HcalLutMetadataRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLutMetadata-> ...";
  std::auto_ptr<HcalLutMetadata> result (new HcalLutMetadata ());

  result->setRctLsb( 0.25 );
  result->setNominalGain( 0.177 );

  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalLutMetadatum item(cell->rawId(),1.0,1,1);
    result->addValues(item);
  }
  return result;
}

std::auto_ptr<HcalDcsValues> 
  HcalHardcodeCalibrations::produceDcsValues (const HcalDcsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceDcsValues-> ...";
  std::auto_ptr<HcalDcsValues> result(new HcalDcsValues);
  return result;
}

std::auto_ptr<HcalDcsMap> HcalHardcodeCalibrations::produceDcsMap (const HcalDcsMapRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceDcsMap-> ...";

  std::auto_ptr<HcalDcsMap> result (new HcalDcsMap ());
  HcalDbHardcode::makeHardcodeDcsMap(*result);
  return result;
}

std::auto_ptr<HcalRecoParams> HcalHardcodeCalibrations::produceRecoParams (const HcalRecoParamsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceRecoParams-> ...";
  std::auto_ptr<HcalRecoParams> result (new HcalRecoParams ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalRecoParam item = HcalDbHardcode::makeRecoParam (*cell);
    result->addValues(item);
  }
  return result;
}
std::auto_ptr<HcalTimingParams> HcalHardcodeCalibrations::produceTimingParams (const HcalTimingParamsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTimingParams-> ...";
  std::auto_ptr<HcalTimingParams> result (new HcalTimingParams ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalTimingParam item = HcalDbHardcode::makeTimingParam (*cell);
    result->addValues(item);
  }
  return result;
}
std::auto_ptr<HcalLongRecoParams> HcalHardcodeCalibrations::produceLongRecoParams (const HcalLongRecoParamsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLongRecoParams-> ...";
  std::auto_ptr<HcalLongRecoParams> result (new HcalLongRecoParams ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  std::vector <unsigned int> mSignal; 
  mSignal.push_back(4); 
  mSignal.push_back(5); 
  mSignal.push_back(6);
  std::vector <unsigned int> mNoise;  
  mNoise.push_back(1);  
  mNoise.push_back(2);  
  mNoise.push_back(3);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    if (cell->isHcalZDCDetId())
      {
	HcalLongRecoParam item(cell->rawId(),mSignal,mNoise);
	result->addValues(item);
      }
  }
  return result;
}

std::auto_ptr<HcalMCParams> HcalHardcodeCalibrations::produceMCParams (const HcalMCParamsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceMCParams-> ...";
  std::auto_ptr<HcalMCParams> result (new HcalMCParams ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalMCParam item(cell->rawId(),0);
    result->addValues(item);
  }
  return result;
}


std::auto_ptr<HcalFlagHFDigiTimeParams> HcalHardcodeCalibrations::produceFlagHFDigiTimeParams (const HcalFlagHFDigiTimeParamsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceFlagHFDigiTimeParams-> ...";
  std::auto_ptr<HcalFlagHFDigiTimeParams> result (new HcalFlagHFDigiTimeParams ());
  if (mode_!=HcalTopology::md_LHC) result->setSlowMode(true);
  std::vector <HcalGenericDetId> cells = allCells(mode_);
  
  std::vector<double> coef;
  coef.push_back(0.93);
  coef.push_back(-0.38275);
  coef.push_back(-0.012667);

  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalFlagHFDigiTimeParam item(cell->rawId(),
				 1, //firstsample
				 3, // samplestoadd
				 2, //expectedpeak
				 40., // min energy threshold
				 coef // coefficients
				 );
    result->addValues(item);
  }
  return result;
} // produceFlagHFDigiTimeParams;
