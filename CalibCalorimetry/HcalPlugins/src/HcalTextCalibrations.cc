// -*- C++ -*-
// Original Author:  Fedor Ratnikov
//
//

#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ValidityInterval.h"

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
  for (; request != data.end (); ++request) {
    std::string objectName = request->getParameter<std::string> ("object");
    edm::FileInPath fp = request->getParameter<edm::FileInPath>("file");
    mInputs [objectName] = fp.fullPath();
//   std::cout << objectName << " with file " << fp.fullPath() << std::endl;
    if (objectName == "Pedestals") {
      setWhatProduced (this, &HcalTextCalibrations::producePedestals);
      findingRecord <HcalPedestalsRcd> ();
    }
    else if (objectName == "PedestalWidths") {
      setWhatProduced (this, &HcalTextCalibrations::producePedestalWidths);
      findingRecord <HcalPedestalWidthsRcd> ();
    }
    else if (objectName == "EffectivePedestals") {
      setWhatProduced (this, &HcalTextCalibrations::produceEffectivePedestals, edm::es::Label("effective"));
      findingRecord <HcalPedestalsRcd> ();
    }
    else if (objectName == "EffectivePedestalWidths") {
      setWhatProduced (this, &HcalTextCalibrations::produceEffectivePedestalWidths, edm::es::Label("effective"));
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
    else if (objectName == "QIETypes") {
      setWhatProduced (this, &HcalTextCalibrations::produceQIETypes);
      findingRecord <HcalQIETypesRcd> ();
    }
    else if (objectName == "ChannelQuality") {
      setWhatProduced (this, &HcalTextCalibrations::produceChannelQuality, edm::es::Label("withTopo"));
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
    else if (objectName == "FrontEndMap") {
      setWhatProduced (this, &HcalTextCalibrations::produceFrontEndMap);
      findingRecord <HcalFrontEndMapRcd> ();
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
    else if (objectName == "ZDCLowGainFractions") {
      setWhatProduced (this, &HcalTextCalibrations::produceZDCLowGainFractions);
      findingRecord <HcalZDCLowGainFractionsRcd> ();
    }
    else if (objectName == "MCParams") {
      setWhatProduced (this, &HcalTextCalibrations::produceMCParams);
      findingRecord <HcalMCParamsRcd> ();
    }
    else if (objectName == "FlagHFDigiTimeParams") {
      setWhatProduced (this, &HcalTextCalibrations::produceFlagHFDigiTimeParams);
      findingRecord <HcalFlagHFDigiTimeParamsRcd> ();
    }
    else if (objectName == "SiPMParameters") {
      setWhatProduced (this, &HcalTextCalibrations::produceSiPMParameters);
      findingRecord <HcalSiPMParametersRcd> ();
    }
    else if (objectName == "SiPMCharacteristics") {
      setWhatProduced (this, &HcalTextCalibrations::produceSiPMCharacteristics);
      findingRecord <HcalSiPMCharacteristicsRcd> ();
    }
    else if (objectName == "TPChannelParameters") {
      setWhatProduced (this, &HcalTextCalibrations::produceTPChannelParameters);
      findingRecord <HcalTPChannelParametersRcd> ();
    }
    else if (objectName == "TPParameters") {
      setWhatProduced (this, &HcalTextCalibrations::produceTPParameters);
      findingRecord <HcalTPParametersRcd> ();
    }
    else {
      std::cerr << "HcalTextCalibrations-> Unknown object name '" << objectName 
		<< "', known names are: "
		<< "Pedestals PedestalWidths Gains GainWidths QIEData QIETypes ChannelQuality ElectronicsMap "
		<< "FrontEndMap ZSThresholds RespCorrs LUTCorrs PFCorrs TimeCorrs L1TriggerObjects "
		<< "ValidationCorrs LutMetadata DcsValues DcsMap "
		<< "RecoParams LongRecoParams ZDCLowGainFraction FlagHFDigiTimeParams MCParams "
		<< "SiPMParameters SiPMCharacteristics TPChannelParameters TPParameters" << std::endl;
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

template <class T, template <class> class F>
std::unique_ptr<T> produce_impl (const std::string& fFile, const HcalTopology* topo=nullptr) {
  std::ifstream inStream (fFile.c_str ());
  if (!inStream.good ()) {
    std::cerr << "HcalTextCalibrations-> Unable to open file '" << fFile << "'" << std::endl;
    throw cms::Exception("FileNotFound") << "Unable to open '" << fFile << "'" << std::endl;
  }
  auto result = F<T>(topo)(inStream);
  if (!result) {
    std::cerr << "HcalTextCalibrations-> Can not read object from file '" << fFile << "'" << std::endl;
    throw cms::Exception("ReadError") << "Can not read object from file '" << fFile << "'" << std::endl;
  }
  return result;
}
template <class T> std::unique_ptr<T> get_impl (const std::string& fFile) { return produce_impl<T,HcalTextCalibrations::CheckGetObject>(fFile); }
template <class T> std::unique_ptr<T> get_impl_topo (const std::string& fFile, const HcalTopology* topo) { return produce_impl<T,HcalTextCalibrations::CheckGetObjectTopo>(fFile,topo); }
template <class T> std::unique_ptr<T> create_impl (const std::string& fFile) { return produce_impl<T,HcalTextCalibrations::CheckCreateObject>(fFile); }


std::unique_ptr<HcalPedestals> HcalTextCalibrations::producePedestals (const HcalPedestalsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  return get_impl_topo<HcalPedestals> (mInputs ["Pedestals"],topo);
}

std::unique_ptr<HcalPedestalWidths> HcalTextCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalPedestalWidths> (mInputs ["PedestalWidths"],topo);
}

std::unique_ptr<HcalPedestals> HcalTextCalibrations::produceEffectivePedestals (const HcalPedestalsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  return get_impl_topo<HcalPedestals> (mInputs ["EffectivePedestals"],topo);
}

std::unique_ptr<HcalPedestalWidths> HcalTextCalibrations::produceEffectivePedestalWidths (const HcalPedestalWidthsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalPedestalWidths> (mInputs ["EffectivePedestalWidths"],topo);
}

std::unique_ptr<HcalGains> HcalTextCalibrations::produceGains (const HcalGainsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalGains> (mInputs ["Gains"],topo);
}

std::unique_ptr<HcalGainWidths> HcalTextCalibrations::produceGainWidths (const HcalGainWidthsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalGainWidths> (mInputs ["GainWidths"],topo);
}

std::unique_ptr<HcalQIEData> HcalTextCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalQIEData> (mInputs ["QIEData"],topo);
}

std::unique_ptr<HcalQIETypes> HcalTextCalibrations::produceQIETypes (const HcalQIETypesRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalQIETypes> (mInputs ["QIETypes"],topo);
}

std::unique_ptr<HcalChannelQuality> HcalTextCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalChannelQuality> (mInputs ["ChannelQuality"],topo);
}

std::unique_ptr<HcalZSThresholds> HcalTextCalibrations::produceZSThresholds (const HcalZSThresholdsRcd& rcd) {  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalZSThresholds> (mInputs ["ZSThresholds"],topo);
}

std::unique_ptr<HcalRespCorrs> HcalTextCalibrations::produceRespCorrs (const HcalRespCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalRespCorrs> (mInputs ["RespCorrs"],topo);
}

std::unique_ptr<HcalLUTCorrs> HcalTextCalibrations::produceLUTCorrs (const HcalLUTCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalLUTCorrs> (mInputs ["LUTCorrs"],topo);
}

std::unique_ptr<HcalPFCorrs> HcalTextCalibrations::producePFCorrs (const HcalPFCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalPFCorrs> (mInputs ["PFCorrs"],topo);
}

std::unique_ptr<HcalTimeCorrs> HcalTextCalibrations::produceTimeCorrs (const HcalTimeCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalTimeCorrs> (mInputs ["TimeCorrs"],topo);
}

std::unique_ptr<HcalL1TriggerObjects> HcalTextCalibrations::produceL1TriggerObjects (const HcalL1TriggerObjectsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalL1TriggerObjects> (mInputs ["L1TriggerObjects"],topo);
}

std::unique_ptr<HcalElectronicsMap> HcalTextCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  return create_impl<HcalElectronicsMap> (mInputs ["ElectronicsMap"]);
}

std::unique_ptr<HcalFrontEndMap> HcalTextCalibrations::produceFrontEndMap (const HcalFrontEndMapRcd& rcd) {
  return create_impl<HcalFrontEndMap> (mInputs ["FrontEndMap"]);
}

std::unique_ptr<HcalValidationCorrs> HcalTextCalibrations::produceValidationCorrs (const HcalValidationCorrsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalValidationCorrs> (mInputs ["ValidationCorrs"],topo);
}

std::unique_ptr<HcalLutMetadata> HcalTextCalibrations::produceLutMetadata (const HcalLutMetadataRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalLutMetadata> (mInputs ["LutMetadata"],topo);
}

std::unique_ptr<HcalDcsValues>
  HcalTextCalibrations::produceDcsValues(HcalDcsRcd const & rcd) {
  return get_impl<HcalDcsValues> (mInputs ["DcsValues"]);
}

std::unique_ptr<HcalDcsMap> HcalTextCalibrations::produceDcsMap (const HcalDcsMapRcd& rcd) {
  return create_impl<HcalDcsMap> (mInputs ["DcsMap"]);
}

std::unique_ptr<HcalRecoParams> HcalTextCalibrations::produceRecoParams (const HcalRecoParamsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalRecoParams> (mInputs ["RecoParams"],topo);
}

std::unique_ptr<HcalLongRecoParams> HcalTextCalibrations::produceLongRecoParams (const HcalLongRecoParamsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalLongRecoParams> (mInputs ["LongRecoParams"],topo);
}

std::unique_ptr<HcalZDCLowGainFractions> HcalTextCalibrations::produceZDCLowGainFractions (const HcalZDCLowGainFractionsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalZDCLowGainFractions> (mInputs ["ZDCLowGainFractions"],topo);
}

std::unique_ptr<HcalTimingParams> HcalTextCalibrations::produceTimingParams (const HcalTimingParamsRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalTimingParams> (mInputs ["TimingParams"],topo);
}
std::unique_ptr<HcalMCParams> HcalTextCalibrations::produceMCParams (const HcalMCParamsRcd& rcd) {  
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalMCParams> (mInputs ["MCParams"],topo);
}

std::unique_ptr<HcalFlagHFDigiTimeParams> HcalTextCalibrations::produceFlagHFDigiTimeParams (const HcalFlagHFDigiTimeParamsRcd& rcd) {  
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalFlagHFDigiTimeParams> (mInputs ["FlagHFDigiTimeParams"],topo);
}

std::unique_ptr<HcalSiPMParameters> HcalTextCalibrations::produceSiPMParameters (const HcalSiPMParametersRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalSiPMParameters> (mInputs ["SiPMParameters"],topo);
}

std::unique_ptr<HcalSiPMCharacteristics> HcalTextCalibrations::produceSiPMCharacteristics (const HcalSiPMCharacteristicsRcd& rcd) {
  return create_impl<HcalSiPMCharacteristics> (mInputs ["SiPMCharacteristics"]);
}

std::unique_ptr<HcalTPChannelParameters> HcalTextCalibrations::produceTPChannelParameters (const HcalTPChannelParametersRcd& rcd) {
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  return get_impl_topo<HcalTPChannelParameters> (mInputs ["TPChannelParameters"],topo);
}

std::unique_ptr<HcalTPParameters> HcalTextCalibrations::produceTPParameters (const HcalTPParametersRcd& rcd) {
  return get_impl<HcalTPParameters> (mInputs ["TPParameters"]);
}
