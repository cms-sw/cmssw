// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// $Id: HcalElectronicsMappingReader.cc,v 1.1 2005/10/25 17:55:39 fedor Exp $
//
//

#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <fstream>
#include <string>

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"

#include "HcalElectronicsMappingReader.h"

namespace {
  std::vector <std::string> splitString (const std::string& fLine) {
    std::vector <std::string> result;
    int start = 0;
    bool empty = true;
    for (unsigned i = 0; i <= fLine.size (); i++) {
      if (fLine [i] == ' ' || i == fLine.size ()) {
	if (!empty) {
	  std::string item (fLine, start, i-start);
	  result.push_back (item);
	  empty = true;
	}
	start = i+1;
      }
      else {
	if (empty) empty = false;
      }
    }
    return result;
  }
}

HcalElectronicsMappingReader::HcalElectronicsMappingReader ( const edm::ParameterSet& iConfig ) 
  
{
  std::cout << "HcalElectronicsMappingReader::HcalElectronicsMappingReader->..." << std::endl;
  //parsing record parameters

  mMapFile = iConfig.getParameter <std::string> ("file");
  setWhatProduced (this);
  findingRecord <HcalElectronicsMapRcd> ();
}


HcalElectronicsMappingReader::~HcalElectronicsMappingReader()
{
}

void 
HcalElectronicsMappingReader::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  std::string record = iKey.name ();
  std::cout << "HcalElectronicsMappingReader::setIntervalFor-> key: " << record << " time: " << iTime.eventID() << '/' << iTime.time ().value () << std::endl;
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

std::auto_ptr<HcalElectronicsMap> HcalElectronicsMappingReader::produce (const HcalElectronicsMapRcd&) {
  std::cout << "HcalElectronicsMappingReader::produce-> ..." << std::endl;
  HcalElectronicsMap* map = new HcalElectronicsMap ();
  readData (mMapFile, map);
  return std::auto_ptr<HcalElectronicsMap> (map);
}

bool HcalElectronicsMappingReader::readData (const std::string& fInput, HcalElectronicsMap* fObject) {
  char buffer [1024];
  std::ifstream in (fInput.empty () ? "/dev/null" : fInput.c_str());
  while (in.getline(buffer, 1024)) {
    if (buffer [0] == '#') continue; //ignore comment
    std::vector <std::string> items = splitString (std::string (buffer));
    if (items.size () < 12) {
      if (items.size () > 0) {
	std::cerr << "Bad line: " << buffer << "\n line must contain 12 items: i  cr sl tb dcc spigot fiber fiberchan subdet ieta iphi depth" << std::endl;
      }
      continue;
    }
    int crate = atoi (items [1].c_str());
    int slot = atoi (items [2].c_str());
    int top = 1;
    if (items [3] == "b") top = 0;
    int dcc = atoi (items [4].c_str());
    int spigot = atoi (items [5].c_str());
    int fiber = atoi (items [6].c_str());
    int fiberCh = atoi (items [7].c_str());
    HcalSubdetector subdet = HcalBarrel;
    if (items [8] == "HE") subdet = HcalEndcap;
    else if (items [8] == "HF") subdet = HcalForward;
    else if (items [8] == "HT") subdet = HcalTriggerTower;
    int eta = atoi (items [9].c_str());
    int phi = atoi (items [10].c_str());
    int depth = atoi (items [11].c_str());
    
    HcalElectronicsId elId (fiberCh, fiber, spigot, dcc);
    elId.setHTR (crate, slot, top);
    if (subdet == HcalTriggerTower) {
      HcalTrigTowerDetId trigId (eta, phi);
      fObject->mapEId2tId (elId (), trigId.rawId());
    }
    else {
      HcalDetId chId (subdet, eta, phi, depth);
      fObject->mapEId2chId (elId (), chId.rawId());
    }
  }
  fObject->sortByElectronicsId ();
  return true;
}

