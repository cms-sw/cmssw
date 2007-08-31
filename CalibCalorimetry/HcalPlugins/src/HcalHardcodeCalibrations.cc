// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// $Id: HcalHardcodeCalibrations.cc,v 1.10 2007/05/11 14:53:04 mansj Exp $
//
//

#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <map>

#include "FWCore/Framework/interface/ValidityInterval.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HcalHardcodeCalibrations.h"

// class decleration
//

using namespace cms;

namespace {

std::vector<HcalGenericDetId> allCells (bool h2_mode) {
  static std::vector<HcalGenericDetId> result;
  if (result.size () <= 0) {
    HcalTopology hcaltopology(h2_mode);
    for (int eta = -50; eta < 50; eta++) {
      for (int phi = 0; phi < 100; phi++) {
	for (int depth = 1; depth < 5; depth++) {
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
  h2mode_=iConfig.getUntrackedParameter<bool>("H2Mode",false);
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
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPedestal item = HcalDbHardcode::makePedestal (*cell);
    result->addValue (*cell, item.getValues ());
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalPedestalWidths> HcalHardcodeCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePedestalWidths-> ...";
  std::auto_ptr<HcalPedestalWidths> result (new HcalPedestalWidths ());
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPedestalWidth item = HcalDbHardcode::makePedestalWidth (*cell);
    result->setWidth (item);
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalGains> HcalHardcodeCalibrations::produceGains (const HcalGainsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceGains-> ...";
  std::auto_ptr<HcalGains> result (new HcalGains ());
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalGain item = HcalDbHardcode::makeGain (*cell);
    result->addValue (*cell, item.getValues ());
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalGainWidths> HcalHardcodeCalibrations::produceGainWidths (const HcalGainWidthsRcd&) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceGainWidths-> ...";
  std::auto_ptr<HcalGainWidths> result (new HcalGainWidths ());
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalGainWidth item = HcalDbHardcode::makeGainWidth (*cell);
    result->addValue (*cell, item.getValues ());
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalQIEData> HcalHardcodeCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceQIEData-> ...";
  std::auto_ptr<HcalQIEData> result (new HcalQIEData ());
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalQIECoder coder = HcalDbHardcode::makeQIECoder (*cell);
    result->addCoder (*cell, coder);
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalChannelQuality> HcalHardcodeCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceChannelQuality-> ...";
  std::auto_ptr<HcalChannelQuality> result (new HcalChannelQuality ());
  std::vector <HcalGenericDetId> cells = allCells(h2mode_);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    result->setChannel (cell->rawId (), HcalChannelQuality::GOOD);
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalElectronicsMap> HcalHardcodeCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceElectronicsMap-> ...";

  std::auto_ptr<HcalElectronicsMap> result (new HcalElectronicsMap ());
  HcalDbHardcode::makeHardcodeMap(*result);
  return result;
}

