#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"

#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "CondFormats/CastorObjects/interface/CastorGains.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidths.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"

#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"

#include "CastorTextCalibrations.h"
//
// class decleration
//

using namespace cms;

CastorTextCalibrations::CastorTextCalibrations ( const edm::ParameterSet& iConfig ) 
  
{
  //parsing parameters
  std::vector<edm::ParameterSet> data = iConfig.getParameter<std::vector<edm::ParameterSet> >("input");
  std::vector<edm::ParameterSet>::iterator request = data.begin ();
  for (; request != data.end (); request++) {
    std::string objectName = request->getParameter<std::string> ("object");
    edm::FileInPath fp = request->getParameter<edm::FileInPath>("file");
    mInputs [objectName] = fp.fullPath();
    if (objectName == "Pedestals") {
      setWhatProduced (this, &CastorTextCalibrations::producePedestals);
      findingRecord <CastorPedestalsRcd> ();
    }
    else if (objectName == "PedestalWidths") {
      setWhatProduced (this, &CastorTextCalibrations::producePedestalWidths);
      findingRecord <CastorPedestalWidthsRcd> ();
    }
    else if (objectName == "Gains") {
      setWhatProduced (this, &CastorTextCalibrations::produceGains);
      findingRecord <CastorGainsRcd> ();
    }
    else if (objectName == "GainWidths") {
      setWhatProduced (this, &CastorTextCalibrations::produceGainWidths);
      findingRecord <CastorGainWidthsRcd> ();
    }
    else if (objectName == "QIEData") {
      setWhatProduced (this, &CastorTextCalibrations::produceQIEData);
      findingRecord <CastorQIEDataRcd> ();
    }
    else if (objectName == "ChannelQuality") {
      setWhatProduced (this, &CastorTextCalibrations::produceChannelQuality);
      findingRecord <CastorChannelQualityRcd> ();
    }
    else if (objectName == "ElectronicsMap") {
      setWhatProduced (this, &CastorTextCalibrations::produceElectronicsMap);
      findingRecord <CastorElectronicsMapRcd> ();
    }
    else if (objectName == "RecoParams") {
      setWhatProduced (this, &CastorTextCalibrations::produceRecoParams);
      findingRecord <CastorRecoParamsRcd> ();
    }
    else if (objectName == "SaturationCorrs") {
      setWhatProduced (this, &CastorTextCalibrations::produceSaturationCorrs);
      findingRecord <CastorSaturationCorrsRcd> ();
    }
    else {
      std::cerr << "CastorTextCalibrations-> Unknown object name '" << objectName 
		<< "', known names are: "
		<< "Pedestals PedestalWidths Gains GainWidths QIEData ChannelQuality ElectronicsMap RecoParams SaturationCorrs"
		<< std::endl;
    }
  }
  //  setWhatProduced(this);
}


CastorTextCalibrations::~CastorTextCalibrations()
{
}


//
// member functions
//
void 
CastorTextCalibrations::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  std::string record = iKey.name ();
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

template <class T>
std::auto_ptr<T> produce_impl (const std::string& fFile) {
  std::auto_ptr<T> result (new T ());
  std::ifstream inStream (fFile.c_str ());
  if (!inStream.good ()) {
    std::cerr << "CastorTextCalibrations-> Unable to open file '" << fFile << "'" << std::endl;
    throw cms::Exception("FileNotFound") << "Unable to open '" << fFile << "'" << std::endl;
  }
  if (!CastorDbASCIIIO::getObject (inStream, &*result)) {
    std::cerr << "CastorTextCalibrations-> Can not read object from file '" << fFile << "'" << std::endl;
    throw cms::Exception("ReadError") << "Can not read object from file '" << fFile << "'" << std::endl;
  }
  return result;
}


std::auto_ptr<CastorPedestals> CastorTextCalibrations::producePedestals (const CastorPedestalsRcd&) {
  return produce_impl<CastorPedestals> (mInputs ["Pedestals"]);
}

std::auto_ptr<CastorPedestalWidths> CastorTextCalibrations::producePedestalWidths (const CastorPedestalWidthsRcd&) {
  return produce_impl<CastorPedestalWidths> (mInputs ["PedestalWidths"]);
}

std::auto_ptr<CastorGains> CastorTextCalibrations::produceGains (const CastorGainsRcd&) {
  return produce_impl<CastorGains> (mInputs ["Gains"]);
}

std::auto_ptr<CastorGainWidths> CastorTextCalibrations::produceGainWidths (const CastorGainWidthsRcd&) {
  return produce_impl<CastorGainWidths> (mInputs ["GainWidths"]);
}

std::auto_ptr<CastorQIEData> CastorTextCalibrations::produceQIEData (const CastorQIEDataRcd& rcd) {
  return produce_impl<CastorQIEData> (mInputs ["QIEData"]);
}

std::auto_ptr<CastorChannelQuality> CastorTextCalibrations::produceChannelQuality (const CastorChannelQualityRcd& rcd) {
  return produce_impl<CastorChannelQuality> (mInputs ["ChannelQuality"]);
}

std::auto_ptr<CastorElectronicsMap> CastorTextCalibrations::produceElectronicsMap (const CastorElectronicsMapRcd& rcd) {
  return produce_impl<CastorElectronicsMap> (mInputs ["ElectronicsMap"]);
}

std::auto_ptr<CastorRecoParams> CastorTextCalibrations::produceRecoParams (const CastorRecoParamsRcd& rcd) {
  return produce_impl<CastorRecoParams> (mInputs ["RecoParams"]);
}

std::auto_ptr<CastorSaturationCorrs> CastorTextCalibrations::produceSaturationCorrs (const CastorSaturationCorrsRcd& rcd) {
  return produce_impl<CastorSaturationCorrs> (mInputs ["SaturationCorrs"]);
}
