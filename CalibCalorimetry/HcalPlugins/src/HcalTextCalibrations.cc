// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// $Id: HcalHardcodeCalibrations.cc,v 1.2 2005/10/28 01:30:47 fedor Exp $
//
//

#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDetIdDb.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
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


#include "HcalTextCalibrations.h"
//
// class decleration
//

using namespace cms;

HcalTextCalibrations::HcalTextCalibrations ( const edm::ParameterSet& iConfig ) 
  
{
  std::cout << "HcalTextCalibrations::HcalTextCalibrations->..." << std::endl;
  //parsing parameters
  std::vector<edm::ParameterSet> data = iConfig.getParameter<std::vector<edm::ParameterSet> >("input");
  std::vector<edm::ParameterSet>::iterator request = data.begin ();
  for (; request != data.end (); request++) {
    std::string objectName = request->getParameter<std::string> ("object");
    std::string file = request->getParameter<std::string> ("file");
    mInputs [objectName] = file;
    if (objectName == "Pedestals") {
      setWhatProduced (this, &HcalTextCalibrations::producePedestals);
      findingRecord <HcalPedestalsRcd> ();
    }
    else if (objectName == "PedestalWidths") {
      setWhatProduced (this, &HcalTextCalibrations::producePedestalWidths);
      findingRecord <HcalPedestalWidthsRcd> ();
    }
    else if (objectName == "Gains") {
      setWhatProduced (this, &HcalTextCalibrations::produceGains);
      findingRecord <HcalGainsRcd> ();
    }
    else if (objectName == "GainWidths") {
      setWhatProduced (this, &HcalTextCalibrations::produceGainWidths);
      findingRecord <HcalGainWidthsRcd> ();
    }
    else if (objectName == "QIEShape") {
      setWhatProduced (this, &HcalTextCalibrations::produceQIEShape);
      findingRecord <HcalQIEShapeRcd> ();
    }
    else if (objectName == "QIEData") {
      setWhatProduced (this, &HcalTextCalibrations::produceQIEData);
      findingRecord <HcalQIEDataRcd> ();
    }
    else if (objectName == "ChannelQuality") {
      setWhatProduced (this, &HcalTextCalibrations::produceChannelQuality);
      findingRecord <HcalChannelQualityRcd> ();
    }
    else if (objectName == "ElectronicsMap") {
      setWhatProduced (this, &HcalTextCalibrations::produceElectronicsMap);
      findingRecord <HcalElectronicsMapRcd> ();
    }
    else {
      std::cerr << "HcalTextCalibrations-> Unknown object name '" << objectName 
		<< "', known names are: "
		<< "Pedestals PedestalWidths Gains GainWidths QIEShape QIEData ChannelQuality ElectronicsMap"
		<< std::endl;
    }
  }
  //  setWhatProduced(this);
}


HcalTextCalibrations::~HcalTextCalibrations()
{
}


//
// member functions
//
void 
HcalTextCalibrations::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  std::string record = iKey.name ();
  std::cout << "HcalTextCalibrations::setIntervalFor-> key: " << record << " time: " << iTime.eventID() << '/' << iTime.time ().value () << std::endl;
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

template <class T>
std::auto_ptr<T> produce_impl (const std::string& fFile) {
  std::auto_ptr<T> result (new T ());
  std::ifstream inStream (fFile.c_str ());
  if (!inStream.good ()) {
    std::cerr << "HcalTextCalibrations-> Unable to open file '" << fFile << "'" << std::endl;
    throw cms::Exception("FileNotFound") << "Unable to open '" << fFile << "'" << std::endl;
  }
  if (!HcalDbASCIIIO::getObject (inStream, &*result)) {
    std::cerr << "HcalTextCalibrations-> Can not read object from file '" << fFile << "'" << std::endl;
    throw cms::Exception("ReadError") << "Can not read object from file '" << fFile << "'" << std::endl;
  }
  return result;
}


std::auto_ptr<HcalPedestals> HcalTextCalibrations::producePedestals (const HcalPedestalsRcd&) {
  std::cout << "HcalTextCalibrations::producePedestals-> ..." << std::endl;
  return produce_impl<HcalPedestals> (mInputs ["Pedestals"]);
}

std::auto_ptr<HcalPedestalWidths> HcalTextCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd&) {
  std::cout << "HcalTextCalibrations::producePedestalWidths-> ..." << std::endl;
  return produce_impl<HcalPedestalWidths> (mInputs ["PedestalWidths"]);
}

std::auto_ptr<HcalGains> HcalTextCalibrations::produceGains (const HcalGainsRcd&) {
  std::cout << "HcalTextCalibrations::produceGains-> ..." << std::endl;
  return produce_impl<HcalGains> (mInputs ["Gains"]);
}

std::auto_ptr<HcalGainWidths> HcalTextCalibrations::produceGainWidths (const HcalGainWidthsRcd&) {
  std::cout << "HcalTextCalibrations::produceGainWidths-> ..." << std::endl;
  return produce_impl<HcalGainWidths> (mInputs ["GainWidths"]);
}

std::auto_ptr<HcalQIEShape> HcalTextCalibrations::produceQIEShape (const HcalQIEShapeRcd&) {
  std::cout << "HcalTextCalibrations::produceQIEShape-> ..." << std::endl;
  return produce_impl<HcalQIEShape> (mInputs ["QIEShape"]);
}

std::auto_ptr<HcalQIEData> HcalTextCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  std::cout << "HcalTextCalibrations::produceQIEData-> ..." << std::endl;
  return produce_impl<HcalQIEData> (mInputs ["QIEData"]);
}

std::auto_ptr<HcalChannelQuality> HcalTextCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  std::cout << "HcalTextCalibrations::produceChannelQuality-> ..." << std::endl;
  return produce_impl<HcalChannelQuality> (mInputs ["ChannelQuality"]);
}

std::auto_ptr<HcalElectronicsMap> HcalTextCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  std::cout << "HcalTextCalibrations::produceElectronicsMap-> ..." << std::endl;
  return produce_impl<HcalElectronicsMap> (mInputs ["ElectronicsMap"]);
}

