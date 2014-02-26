// -*- C++ -*-
// Original Author:  Fedor Ratnikov
// $Id: HcalTextCalibrations.cc,v 1.22 2012/11/12 20:42:39 dlange Exp $
//
//

#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "CondFormats/DataRecord/interface/HcalAllRcds.h"

#include "HcalTextCalibrations.h"
//
// class declaration
//

using namespace cms;

HcalTextCalibrations::HcalTextCalibrations ( const edm::ParameterSet& iConfig ) 
  
{
  //parsing parameters
  std::vector<edm::ParameterSet> data = iConfig.getParameter<std::vector<edm::ParameterSet> >("input");
  std::vector<edm::ParameterSet>::iterator request = data.begin ();
  for (; request != data.end (); request++) {
    std::string objectName = request->getParameter<std::string> ("object");
    edm::FileInPath fp = request->getParameter<edm::FileInPath>("file");
    mInputs [objectName] = fp.fullPath();
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
    else if (objectName == "QIEData") {
      setWhatProduced (this, &HcalTextCalibrations::produceQIEData);
      findingRecord <HcalQIEDataRcd> ();
    }
    else if (objectName == "ChannelQuality") {
      setWhatProduced (this, &HcalTextCalibrations::produceChannelQuality);
      findingRecord <HcalChannelQualityRcd> ();
    }
    else if (objectName == "ZSThresholds") {
      setWhatProduced (this, &HcalTextCalibrations::produceZSThresholds);
      findingRecord <HcalZSThresholdsRcd> ();
    }
    else if (objectName == "RespCorrs") {
      setWhatProduced (this, &HcalTextCalibrations::produceRespCorrs);
      findingRecord <HcalRespCorrsRcd> ();
    }
    else if (objectName == "LUTCorrs") {
      setWhatProduced (this, &HcalTextCalibrations::produceLUTCorrs);
      findingRecord <HcalLUTCorrsRcd> ();
    }
    else if (objectName == "PFCorrs") {
      setWhatProduced (this, &HcalTextCalibrations::producePFCorrs);
      findingRecord <HcalPFCorrsRcd> ();
    }
    else if (objectName == "TimeCorrs") {
      setWhatProduced (this, &HcalTextCalibrations::produceTimeCorrs);
      findingRecord <HcalTimeCorrsRcd> ();
    }
    else if (objectName == "L1TriggerObjects") {
      setWhatProduced (this, &HcalTextCalibrations::produceL1TriggerObjects);
      findingRecord <HcalL1TriggerObjectsRcd> ();
    }
    else if (objectName == "ElectronicsMap") {
      setWhatProduced (this, &HcalTextCalibrations::produceElectronicsMap);
      findingRecord <HcalElectronicsMapRcd> ();
    }
    else if (objectName == "ValidationCorrs") {
      setWhatProduced (this, &HcalTextCalibrations::produceValidationCorrs);
      findingRecord <HcalValidationCorrsRcd> ();
    }
    else if (objectName == "LutMetadata") {
      setWhatProduced (this, &HcalTextCalibrations::produceLutMetadata);
      findingRecord <HcalLutMetadataRcd> ();
    }
    else if (objectName == "DcsValues") {
      setWhatProduced (this, &HcalTextCalibrations::produceDcsValues);
      findingRecord <HcalDcsRcd> ();
    }
    else if (objectName == "DcsMap") {
      setWhatProduced (this, &HcalTextCalibrations::produceDcsMap);
      findingRecord <HcalDcsMapRcd> ();
    }
    else if (objectName == "CholeskyMatrices") {
      setWhatProduced (this, &HcalTextCalibrations::produceCholeskyMatrices);
      findingRecord <HcalCholeskyMatricesRcd> ();
    }
    else if (objectName == "CovarianceMatrices") {
      setWhatProduced (this, &HcalTextCalibrations::produceCovarianceMatrices);
      findingRecord <HcalCovarianceMatricesRcd> ();
    }
    else if (objectName == "RecoParams") {
      setWhatProduced (this, &HcalTextCalibrations::produceRecoParams);
      findingRecord <HcalRecoParamsRcd> ();
    }
    else if (objectName == "TimingParams") {
      setWhatProduced (this, &HcalTextCalibrations::produceTimingParams);
      findingRecord <HcalTimingParamsRcd> ();
    }
    else if (objectName == "LongRecoParams") {
      setWhatProduced (this, &HcalTextCalibrations::produceLongRecoParams);
      findingRecord <HcalLongRecoParamsRcd> ();
    }
    else if (objectName == "MCParams") {
      setWhatProduced (this, &HcalTextCalibrations::produceMCParams);
      findingRecord <HcalMCParamsRcd> ();
    }
    else if (objectName == "FlagHFDigiTimeParams") {
      setWhatProduced (this, &HcalTextCalibrations::produceFlagHFDigiTimeParams);
      findingRecord <HcalFlagHFDigiTimeParamsRcd> ();
    }
    else {
      std::cerr << "HcalTextCalibrations-> Unknown object name '" << objectName 
		<< "', known names are: "
		<< "Pedestals PedestalWidths Gains GainWidths QIEData ChannelQuality ElectronicsMap "
		<< "ZSThresholds RespCorrs LUTCorrs PFCorrs TimeCorrs L1TriggerObjects "
		<< "ValidationCorrs LutMetadata DcsValues DcsMap CholeskyMatrices CovarianceMatrices "
		<< "RecoParams LongRecoParams FlagHFDigiTimeParams MCParams "
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
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

template <class T>
std::auto_ptr<T> produce_impl (const HcalTopology* topo, const std::string& fFile) {
  std::auto_ptr<T> result (new T (topo));
  //  std::auto_ptr<T> result;
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

template <class T>
std::auto_ptr<T> produce_impl (const std::string& fFile) {
  std::auto_ptr<T> result (new T ());
  //  std::auto_ptr<T> result;
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



std::auto_ptr<HcalPedestals> HcalTextCalibrations::producePedestals (const HcalPedestalsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  return produce_impl<HcalPedestals> (topo,mInputs ["Pedestals"]);
}

std::auto_ptr<HcalPedestalWidths> HcalTextCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalPedestalWidths> (topo,mInputs ["PedestalWidths"]);
}

std::auto_ptr<HcalGains> HcalTextCalibrations::produceGains (const HcalGainsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalGains> (topo,mInputs ["Gains"]);
}

std::auto_ptr<HcalGainWidths> HcalTextCalibrations::produceGainWidths (const HcalGainWidthsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalGainWidths> (topo,mInputs ["GainWidths"]);
}

std::auto_ptr<HcalQIEData> HcalTextCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalQIEData> (topo,mInputs ["QIEData"]);
}

std::auto_ptr<HcalChannelQuality> HcalTextCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalChannelQuality> (topo,mInputs ["ChannelQuality"]);
}

std::auto_ptr<HcalZSThresholds> HcalTextCalibrations::produceZSThresholds (const HcalZSThresholdsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalZSThresholds> (topo,mInputs ["ZSThresholds"]);
}

std::auto_ptr<HcalRespCorrs> HcalTextCalibrations::produceRespCorrs (const HcalRespCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalRespCorrs> (topo,mInputs ["RespCorrs"]);
}

std::auto_ptr<HcalLUTCorrs> HcalTextCalibrations::produceLUTCorrs (const HcalLUTCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalLUTCorrs> (topo,mInputs ["LUTCorrs"]);
}

std::auto_ptr<HcalPFCorrs> HcalTextCalibrations::producePFCorrs (const HcalPFCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalPFCorrs> (topo,mInputs ["PFCorrs"]);
}

std::auto_ptr<HcalTimeCorrs> HcalTextCalibrations::produceTimeCorrs (const HcalTimeCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalTimeCorrs> (topo,mInputs ["TimeCorrs"]);
}

std::auto_ptr<HcalL1TriggerObjects> HcalTextCalibrations::produceL1TriggerObjects (const HcalL1TriggerObjectsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalL1TriggerObjects> (topo,mInputs ["L1TriggerObjects"]);
}

std::auto_ptr<HcalElectronicsMap> HcalTextCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  return produce_impl<HcalElectronicsMap> (mInputs ["ElectronicsMap"]);
}

std::auto_ptr<HcalValidationCorrs> HcalTextCalibrations::produceValidationCorrs (const HcalValidationCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalValidationCorrs> (topo,mInputs ["ValidationCorrs"]);
}

std::auto_ptr<HcalLutMetadata> HcalTextCalibrations::produceLutMetadata (const HcalLutMetadataRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalLutMetadata> (topo,mInputs ["LutMetadata"]);
}

std::auto_ptr<HcalDcsValues>
  HcalTextCalibrations::produceDcsValues(HcalDcsRcd const & rcd) {
  return produce_impl<HcalDcsValues> (mInputs ["DcsValues"]);
}

std::auto_ptr<HcalDcsMap> HcalTextCalibrations::produceDcsMap (const HcalDcsMapRcd& rcd) {
  return produce_impl<HcalDcsMap> (mInputs ["DcsMap"]);
}

std::auto_ptr<HcalCovarianceMatrices> HcalTextCalibrations::produceCovarianceMatrices (const HcalCovarianceMatricesRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalCovarianceMatrices> (topo,mInputs ["CovarianceMatrices"]);
}

std::auto_ptr<HcalCholeskyMatrices> HcalTextCalibrations::produceCholeskyMatrices (const HcalCholeskyMatricesRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalCholeskyMatrices> (topo,mInputs ["CholeskyMatrices"]);
}

std::auto_ptr<HcalRecoParams> HcalTextCalibrations::produceRecoParams (const HcalRecoParamsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalRecoParams> (topo,mInputs ["RecoParams"]);
}

std::auto_ptr<HcalLongRecoParams> HcalTextCalibrations::produceLongRecoParams (const HcalLongRecoParamsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalLongRecoParams> (topo,mInputs ["LongRecoParams"]);
}
std::auto_ptr<HcalTimingParams> HcalTextCalibrations::produceTimingParams (const HcalTimingParamsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalTimingParams> (topo,mInputs ["TimingParams"]);
}
std::auto_ptr<HcalMCParams> HcalTextCalibrations::produceMCParams (const HcalMCParamsRcd& rcd) {  
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalMCParams> (topo,mInputs ["MCParams"]);
}

std::auto_ptr<HcalFlagHFDigiTimeParams> HcalTextCalibrations::produceFlagHFDigiTimeParams (const HcalFlagHFDigiTimeParamsRcd& rcd) {  
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return produce_impl<HcalFlagHFDigiTimeParams> (topo,mInputs ["FlagHFDigiTimeParams"]);
}
