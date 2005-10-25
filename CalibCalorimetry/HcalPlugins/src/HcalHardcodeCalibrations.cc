// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// $Id: PoolDBHcalESSource.cc,v 1.8 2005/09/09 13:15:00 xiezhen Exp $
//
//

#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <map>

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalCannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"

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
  std::vector< std::pair<std::string,std::string> > recordToTag;

  std::vector <std::string> toGet = iConfig.getParameter <std::vector <std::string> > ("toGet");
  for(std::vector <std::string>::iterator objectName = toGet.begin(); objectName != toGet.end(); ++objectName ) {
    if (*objectName == "Pedestals") {
      setWhatProduced (this, &HcalHardcodeCalibrations::producePedestals);
      findingRecord <HcalPedestalsRcd> ();
    }
    if (*objectName == "PedestalWidths") {
      setWhatProduced (this, &HcalHardcodeCalibrations::producePedestalWidths);
      findingRecord <HcalPedestalWidthsRcd> ();
    }
    if (*objectName == "Gains") {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceGains);
      findingRecord <HcalGainsRcd> ();
    }
    if (*objectName == "GainWidths") {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceGainWidths);
      findingRecord <HcalGainWidthsRcd> ();
    }

  }
  //  setWhatProduced(this);
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


