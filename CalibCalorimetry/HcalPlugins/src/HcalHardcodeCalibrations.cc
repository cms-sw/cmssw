// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// $Id: HcalHardcodeCalibrations.cc,v 1.4 2005/12/12 18:57:18 fedor Exp $
//
//

#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <map>

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEShapeRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"


#include "HcalHardcodeCalibrations.h"
//
// class decleration
//

using namespace cms;

namespace {

std::vector<HcalDetId> allCells () {
  static std::vector<HcalDetId> result;
  if (result.size () <= 0) {
    HcalTopology topology;
    for (int eta = -50; eta < 50; eta++) {
      for (int phi = 0; phi < 100; phi++) {
	for (int depth = 1; depth < 5; depth++) {
	  for (int det = 1; det < 5; det++) {
	    HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	    if (topology.valid(cell)) result.push_back (cell);
	  }
	}
      }
    }
  }
  return result;
}

}

HcalHardcodeCalibrations::HcalHardcodeCalibrations ( const edm::ParameterSet& iConfig ) 
  
{
  std::cout << "HcalHardcodeCalibrations::HcalHardcodeCalibrations->..." << std::endl;
  //parsing record parameters
  std::vector <std::string> toGet = iConfig.getParameter <std::vector <std::string> > ("toGet");
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
  std::cout << "HcalHardcodeCalibrations::setIntervalFor-> key: " << record << " time: " << iTime.eventID() << '/' << iTime.time ().value () << std::endl;
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

std::auto_ptr<HcalPedestals> HcalHardcodeCalibrations::producePedestals (const HcalPedestalsRcd&) {
  std::cout << "HcalHardcodeCalibrations::producePedestals-> ..." << std::endl;
  std::auto_ptr<HcalPedestals> result (new HcalPedestals ());
  std::vector <HcalDetId> cells = allCells ();
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPedestal item = HcalDbHardcode::makePedestal (*cell);
    result->addValue (*cell, item.getValues ());
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalPedestalWidths> HcalHardcodeCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd&) {
  std::cout << "HcalHardcodeCalibrations::producePedestalWidths-> ..." << std::endl;
  std::auto_ptr<HcalPedestalWidths> result (new HcalPedestalWidths ());
  std::vector <HcalDetId> cells = allCells ();
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPedestalWidth item = HcalDbHardcode::makePedestalWidth (*cell);
    result->addValue (*cell, item.getValues ());
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalGains> HcalHardcodeCalibrations::produceGains (const HcalGainsRcd&) {
  std::cout << "HcalHardcodeCalibrations::produceGains-> ..." << std::endl;
  std::auto_ptr<HcalGains> result (new HcalGains ());
  std::vector <HcalDetId> cells = allCells ();
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalGain item = HcalDbHardcode::makeGain (*cell);
    result->addValue (*cell, item.getValues ());
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalGainWidths> HcalHardcodeCalibrations::produceGainWidths (const HcalGainWidthsRcd&) {
  std::cout << "HcalHardcodeCalibrations::produceGainWidths-> ..." << std::endl;
  std::auto_ptr<HcalGainWidths> result (new HcalGainWidths ());
  std::vector <HcalDetId> cells = allCells ();
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalGainWidth item = HcalDbHardcode::makeGainWidth (*cell);
    result->addValue (*cell, item.getValues ());
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalQIEData> HcalHardcodeCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  std::cout << "HcalHardcodeCalibrations::produceQIEData-> ..." << std::endl;
  std::auto_ptr<HcalQIEData> result (new HcalQIEData ());
  HcalQIEShape shape = HcalDbHardcode::makeQIEShape ();
  float shapes [32];
  for (unsigned i = 0; i < 32; i++) shapes [i] = shape.lowEdge (i);
  result->setShape (shapes);
  std::vector <HcalDetId> cells = allCells ();
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalQIECoder coder = HcalDbHardcode::makeQIECoder (*cell);
    result->addCoder (*cell, coder);
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalChannelQuality> HcalHardcodeCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  std::cout << "HcalHardcodeCalibrations::produceChannelQuality-> ..." << std::endl;
  std::auto_ptr<HcalChannelQuality> result (new HcalChannelQuality ());
  std::vector <HcalDetId> cells = allCells ();
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    result->setChannel (cell->rawId (), HcalChannelQuality::GOOD);
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalElectronicsMap> HcalHardcodeCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  std::cout << "HcalHardcodeCalibrations::produceElectronicsMap-> Is not implemented..." << std::endl;
  std::auto_ptr<HcalElectronicsMap> result (new HcalElectronicsMap ());
  return result;
}

