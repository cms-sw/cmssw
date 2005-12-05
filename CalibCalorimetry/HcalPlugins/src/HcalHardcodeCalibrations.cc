// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// $Id: HcalHardcodeCalibrations.cc,v 1.2 2005/10/28 01:30:47 fedor Exp $
//
//

#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <map>

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDetIdDb.h"
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


#include "HcalHardcodeCalibrations.h"
//
// class decleration
//

using namespace cms;

namespace {

bool validHcalCell (const HcalDetId& fCell) {
  if (fCell.iphi () <=0)  return false;
  int absEta = abs (fCell.ieta ());
  int phi = fCell.iphi ();
  int depth = fCell.depth ();
  HcalSubdetector det = fCell.subdet ();
  // phi ranges
  if ((absEta >= 40 && phi > 18) ||
      (absEta >= 21 && phi > 36) ||
      phi > 72)   return false;
  if (absEta <= 0)       return false;
  else if (absEta <= 14) return (depth == 1 || depth == 4) && det == HcalBarrel; 
  else if (absEta == 15) return (depth == 1 || depth == 2 || depth == 4) && det == HcalBarrel; 
  else if (absEta == 16) return depth >= 1 && depth <= 2 && det == HcalBarrel || depth == 3 && det == HcalEndcap; 
  else if (absEta == 17) return depth == 1 && det == HcalEndcap; 
  else if (absEta <= 26) return depth >= 1 && depth <= 2 && det == HcalEndcap; 
  else if (absEta <= 28) return depth >= 1 && depth <= 3 && det == HcalEndcap; 
  else if (absEta == 29) return depth >= 1 && depth <= 2 && (det == HcalEndcap || det == HcalForward); 
  else if (absEta <= 41) return depth >= 1 && depth <= 2 && det == HcalForward;
  else return false;
}

std::vector<unsigned long> allCells () {
  static std::vector<unsigned long> result;
  if (result.size () <= 0) {
    for (int eta = -50; eta < 50; eta++) {
      for (int phi = 0; phi < 100; phi++) {
	for (int depth = 1; depth < 5; depth++) {
	  for (int det = 1; det < 5; det++) {
	    HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	    if (validHcalCell(cell)) result.push_back (cell.rawId ());
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
    if ((*objectName == "QIEShape") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceQIEShape);
      findingRecord <HcalQIEShapeRcd> ();
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
  HcalDbServiceHardcode srv;
  std::vector <unsigned long> cells = allCells ();
  for (std::vector <unsigned long>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    result->addValue (*cell, srv.pedestals (*cell));
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalPedestalWidths> HcalHardcodeCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd&) {
  std::cout << "HcalHardcodeCalibrations::producePedestalWidths-> ..." << std::endl;
  std::auto_ptr<HcalPedestalWidths> result (new HcalPedestalWidths ());
  HcalDbServiceHardcode srv;
  std::vector <unsigned long> cells = allCells ();
  for (std::vector <unsigned long>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    result->addValue (*cell, srv.pedestalErrors (*cell));
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalGains> HcalHardcodeCalibrations::produceGains (const HcalGainsRcd&) {
  std::cout << "HcalHardcodeCalibrations::produceGains-> ..." << std::endl;
  std::auto_ptr<HcalGains> result (new HcalGains ());
  HcalDbServiceHardcode srv;
  std::vector <unsigned long> cells = allCells ();
  for (std::vector <unsigned long>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    result->addValue (*cell, srv.gains (*cell));
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalGainWidths> HcalHardcodeCalibrations::produceGainWidths (const HcalGainWidthsRcd&) {
  std::cout << "HcalHardcodeCalibrations::produceGainWidths-> ..." << std::endl;
  std::auto_ptr<HcalGainWidths> result (new HcalGainWidths ());
  HcalDbServiceHardcode srv;
  std::vector <unsigned long> cells = allCells ();
  for (std::vector <unsigned long>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    result->addValue (*cell, srv.gainErrors (*cell));
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalQIEShape> HcalHardcodeCalibrations::produceQIEShape (const HcalQIEShapeRcd&) {
  std::cout << "HcalHardcodeCalibrations::produceQIEShape-> ..." << std::endl;
  std::auto_ptr<HcalQIEShape> result (new HcalQIEShape ());
  
  HcalDbServiceHardcode srv;
  float lowEdges [33];
  for (int i = 0; i < 32; i++) {
    lowEdges [i] = srv.adcShape (i);
  }
  lowEdges [32] = 2 * lowEdges [31] - lowEdges [30];
  result->setLowEdges (lowEdges);
  return result;
}

std::auto_ptr<HcalQIEData> HcalHardcodeCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  std::cout << "HcalHardcodeCalibrations::produceQIEData-> ..." << std::endl;
  std::auto_ptr<HcalQIEData> result (new HcalQIEData ());
  HcalDbServiceHardcode srv;
  std::vector <unsigned long> cells = allCells ();
  for (std::vector <unsigned long>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalDetId id (HcalDetIdDb::HcalDetId (*cell));
    const float* offsetsIn = srv.offsets (id);
    const float* slopesIn = srv.slopes (id);
    float offsetsOut [16];
    float slopesOut [16];
    for (int range = 0; range < 4; range++) {
      for (int cap = 0; cap < 4; cap++) {
	offsetsOut [HcalQIEData::index (range, cap)] = offsetsIn [HcalDbServiceHardcode::index (range, cap)];
	slopesOut [HcalQIEData::index (range, cap)] = slopesIn [HcalDbServiceHardcode::index (range, cap)];
      }
    }
    result->addValue (*cell, offsetsOut, slopesOut);
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalChannelQuality> HcalHardcodeCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  std::cout << "HcalHardcodeCalibrations::produceChannelQuality-> ..." << std::endl;
  std::auto_ptr<HcalChannelQuality> result (new HcalChannelQuality ());
  std::vector <unsigned long> cells = allCells ();
  for (std::vector <unsigned long>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    result->setChannel (*cell, HcalChannelQuality::GOOD);
  }
  result->sort ();
  return result;
}

std::auto_ptr<HcalElectronicsMap> HcalHardcodeCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  std::cout << "HcalHardcodeCalibrations::produceElectronicsMap-> Is not implemented..." << std::endl;
  std::auto_ptr<HcalElectronicsMap> result (new HcalElectronicsMap ());
  return result;
}

