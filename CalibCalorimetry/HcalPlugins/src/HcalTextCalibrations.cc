// -*- C++ -*-
// Original Author:  Fedor Ratnikov
//
//

#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "CondFormats/DataRecord/interface/HcalAllRcds.h"

#include "HcalTextCalibrations.h"
//
// class declaration
//

using namespace cms;

HcalTextCalibrations::HcalTextCalibrations(const edm::ParameterSet& iConfig)

{
  //parsing parameters
  std::vector<edm::ParameterSet> data = iConfig.getParameter<std::vector<edm::ParameterSet> >("input");
  std::vector<edm::ParameterSet>::iterator request = data.begin();
  for (; request != data.end(); ++request) {
    std::string objectName = request->getParameter<std::string>("object");
    edm::FileInPath fp = request->getParameter<edm::FileInPath>("file");
    mInputs[objectName] = fp.fullPath();
    //   std::cout << objectName << " with file " << fp.fullPath() << std::endl;
    if (objectName == "Pedestals") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::producePedestals).consumes();
      findingRecord<HcalPedestalsRcd>();
    } else if (objectName == "PedestalWidths") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::producePedestalWidths).consumes();
      findingRecord<HcalPedestalWidthsRcd>();
    } else if (objectName == "EffectivePedestals") {
      mTokens[objectName] =
          setWhatProduced(this, &HcalTextCalibrations::produceEffectivePedestals, edm::es::Label("effective"))
              .consumes();
      findingRecord<HcalPedestalsRcd>();
    } else if (objectName == "EffectivePedestalWidths") {
      mTokens[objectName] =
          setWhatProduced(this, &HcalTextCalibrations::produceEffectivePedestalWidths, edm::es::Label("effective"))
              .consumes();
      findingRecord<HcalPedestalWidthsRcd>();
    } else if (objectName == "Gains") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceGains).consumes();
      findingRecord<HcalGainsRcd>();
    } else if (objectName == "GainWidths") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceGainWidths).consumes();
      findingRecord<HcalGainWidthsRcd>();
    } else if (objectName == "QIEData") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceQIEData).consumes();
      findingRecord<HcalQIEDataRcd>();
    } else if (objectName == "QIETypes") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceQIETypes).consumes();
      findingRecord<HcalQIETypesRcd>();
    } else if (objectName == "ChannelQuality") {
      mTokens[objectName] =
          setWhatProduced(this, &HcalTextCalibrations::produceChannelQuality, edm::es::Label("withTopo")).consumes();
      findingRecord<HcalChannelQualityRcd>();
    } else if (objectName == "ZSThresholds") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceZSThresholds).consumes();
      findingRecord<HcalZSThresholdsRcd>();
    } else if (objectName == "RespCorrs") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceRespCorrs).consumes();
      findingRecord<HcalRespCorrsRcd>();
    } else if (objectName == "LUTCorrs") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceLUTCorrs).consumes();
      findingRecord<HcalLUTCorrsRcd>();
    } else if (objectName == "PFCorrs") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::producePFCorrs).consumes();
      findingRecord<HcalPFCorrsRcd>();
    } else if (objectName == "TimeCorrs") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceTimeCorrs).consumes();
      findingRecord<HcalTimeCorrsRcd>();
    } else if (objectName == "L1TriggerObjects") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceL1TriggerObjects).consumes();
      findingRecord<HcalL1TriggerObjectsRcd>();
    } else if (objectName == "ElectronicsMap") {
      setWhatProduced(this, &HcalTextCalibrations::produceElectronicsMap);
      findingRecord<HcalElectronicsMapRcd>();
    } else if (objectName == "FrontEndMap") {
      setWhatProduced(this, &HcalTextCalibrations::produceFrontEndMap);
      findingRecord<HcalFrontEndMapRcd>();
    } else if (objectName == "ValidationCorrs") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceValidationCorrs).consumes();
      findingRecord<HcalValidationCorrsRcd>();
    } else if (objectName == "LutMetadata") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceLutMetadata).consumes();
      findingRecord<HcalLutMetadataRcd>();
    } else if (objectName == "DcsValues") {
      setWhatProduced(this, &HcalTextCalibrations::produceDcsValues);
      findingRecord<HcalDcsRcd>();
    } else if (objectName == "DcsMap") {
      setWhatProduced(this, &HcalTextCalibrations::produceDcsMap);
      findingRecord<HcalDcsMapRcd>();
    } else if (objectName == "RecoParams") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceRecoParams).consumes();
      findingRecord<HcalRecoParamsRcd>();
    } else if (objectName == "TimingParams") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceTimingParams).consumes();
      findingRecord<HcalTimingParamsRcd>();
    } else if (objectName == "LongRecoParams") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceLongRecoParams).consumes();
      findingRecord<HcalLongRecoParamsRcd>();
    } else if (objectName == "ZDCLowGainFractions") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceZDCLowGainFractions).consumes();
      findingRecord<HcalZDCLowGainFractionsRcd>();
    } else if (objectName == "MCParams") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceMCParams).consumes();
      findingRecord<HcalMCParamsRcd>();
    } else if (objectName == "FlagHFDigiTimeParams") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceFlagHFDigiTimeParams).consumes();
      findingRecord<HcalFlagHFDigiTimeParamsRcd>();
    } else if (objectName == "SiPMParameters") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceSiPMParameters).consumes();
      findingRecord<HcalSiPMParametersRcd>();
    } else if (objectName == "SiPMCharacteristics") {
      setWhatProduced(this, &HcalTextCalibrations::produceSiPMCharacteristics);
      findingRecord<HcalSiPMCharacteristicsRcd>();
    } else if (objectName == "TPChannelParameters") {
      mTokens[objectName] = setWhatProduced(this, &HcalTextCalibrations::produceTPChannelParameters).consumes();
      findingRecord<HcalTPChannelParametersRcd>();
    } else if (objectName == "TPParameters") {
      setWhatProduced(this, &HcalTextCalibrations::produceTPParameters);
      findingRecord<HcalTPParametersRcd>();
    } else {
      std::cerr << "HcalTextCalibrations-> Unknown object name '" << objectName << "', known names are: "
                << "Pedestals PedestalWidths Gains GainWidths QIEData QIETypes ChannelQuality ElectronicsMap "
                << "FrontEndMap ZSThresholds RespCorrs LUTCorrs PFCorrs TimeCorrs L1TriggerObjects "
                << "ValidationCorrs LutMetadata DcsValues DcsMap "
                << "RecoParams LongRecoParams ZDCLowGainFraction FlagHFDigiTimeParams MCParams "
                << "SiPMParameters SiPMCharacteristics TPChannelParameters TPParameters" << std::endl;
    }
  }
  //  setWhatProduced(this);
}

HcalTextCalibrations::~HcalTextCalibrations() {}

//
// member functions
//
void HcalTextCalibrations::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                          const edm::IOVSyncValue& iTime,
                                          edm::ValidityInterval& oInterval) {
  std::string record = iKey.name();
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());  //infinite
}
namespace {
  template <class T, template <class> class F>
  std::unique_ptr<T> produce_impl(const std::string& fFile, const HcalTopology* topo = nullptr) {
    std::ifstream inStream(fFile.c_str());
    if (!inStream.good()) {
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
  template <class T>
  std::unique_ptr<T> get_impl(const std::string& fFile) {
    return produce_impl<T, HcalTextCalibrations::CheckGetObject>(fFile);
  }
  template <class T>
  std::unique_ptr<T> get_impl_topo(const std::string& fFile, const HcalTopology* topo) {
    return produce_impl<T, HcalTextCalibrations::CheckGetObjectTopo>(fFile, topo);
  }
  template <class T>
  std::unique_ptr<T> create_impl(const std::string& fFile) {
    return produce_impl<T, HcalTextCalibrations::CheckCreateObject>(fFile);
  }
}  // namespace

std::unique_ptr<HcalPedestals> HcalTextCalibrations::producePedestals(const HcalPedestalsRcd& rcd) {
  std::string const n = "Pedestals";
  return get_impl_topo<HcalPedestals>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalPedestalWidths> HcalTextCalibrations::producePedestalWidths(const HcalPedestalWidthsRcd& rcd) {
  std::string const n = "PedestalWidths";
  return get_impl_topo<HcalPedestalWidths>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalPedestals> HcalTextCalibrations::produceEffectivePedestals(const HcalPedestalsRcd& rcd) {
  std::string const n = "EffectivePedestals";
  return get_impl_topo<HcalPedestals>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalPedestalWidths> HcalTextCalibrations::produceEffectivePedestalWidths(
    const HcalPedestalWidthsRcd& rcd) {
  std::string const n = "EffectivePedestalWidths";
  return get_impl_topo<HcalPedestalWidths>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalGains> HcalTextCalibrations::produceGains(const HcalGainsRcd& rcd) {
  std::string const n = "Gains";
  return get_impl_topo<HcalGains>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalGainWidths> HcalTextCalibrations::produceGainWidths(const HcalGainWidthsRcd& rcd) {
  std::string const n = "GainWidths";
  return get_impl_topo<HcalGainWidths>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalQIEData> HcalTextCalibrations::produceQIEData(const HcalQIEDataRcd& rcd) {
  std::string const n = "QIEData";
  return get_impl_topo<HcalQIEData>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalQIETypes> HcalTextCalibrations::produceQIETypes(const HcalQIETypesRcd& rcd) {
  std::string const n = "QIETypes";
  return get_impl_topo<HcalQIETypes>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalChannelQuality> HcalTextCalibrations::produceChannelQuality(const HcalChannelQualityRcd& rcd) {
  std::string const n = "ChannelQuality";
  return get_impl_topo<HcalChannelQuality>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalZSThresholds> HcalTextCalibrations::produceZSThresholds(const HcalZSThresholdsRcd& rcd) {
  std::string const n = "ZSThresholds";
  return get_impl_topo<HcalZSThresholds>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalRespCorrs> HcalTextCalibrations::produceRespCorrs(const HcalRespCorrsRcd& rcd) {
  std::string const n = "RespCorrs";
  return get_impl_topo<HcalRespCorrs>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalLUTCorrs> HcalTextCalibrations::produceLUTCorrs(const HcalLUTCorrsRcd& rcd) {
  std::string const n = "LUTCorrs";
  return get_impl_topo<HcalLUTCorrs>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalPFCorrs> HcalTextCalibrations::producePFCorrs(const HcalPFCorrsRcd& rcd) {
  std::string const n = "PFCorrs";
  return get_impl_topo<HcalPFCorrs>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalTimeCorrs> HcalTextCalibrations::produceTimeCorrs(const HcalTimeCorrsRcd& rcd) {
  std::string const n = "TimeCorrs";
  return get_impl_topo<HcalTimeCorrs>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalL1TriggerObjects> HcalTextCalibrations::produceL1TriggerObjects(const HcalL1TriggerObjectsRcd& rcd) {
  std::string const n = "L1TriggerObjects";
  return get_impl_topo<HcalL1TriggerObjects>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalElectronicsMap> HcalTextCalibrations::produceElectronicsMap(const HcalElectronicsMapRcd& rcd) {
  return create_impl<HcalElectronicsMap>(mInputs["ElectronicsMap"]);
}

std::unique_ptr<HcalFrontEndMap> HcalTextCalibrations::produceFrontEndMap(const HcalFrontEndMapRcd& rcd) {
  return create_impl<HcalFrontEndMap>(mInputs["FrontEndMap"]);
}

std::unique_ptr<HcalValidationCorrs> HcalTextCalibrations::produceValidationCorrs(const HcalValidationCorrsRcd& rcd) {
  std::string const n = "ValidationCorrs";
  return get_impl_topo<HcalValidationCorrs>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalLutMetadata> HcalTextCalibrations::produceLutMetadata(const HcalLutMetadataRcd& rcd) {
  std::string const n = "LutMetadata";
  return get_impl_topo<HcalLutMetadata>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalDcsValues> HcalTextCalibrations::produceDcsValues(HcalDcsRcd const& rcd) {
  return get_impl<HcalDcsValues>(mInputs["DcsValues"]);
}

std::unique_ptr<HcalDcsMap> HcalTextCalibrations::produceDcsMap(const HcalDcsMapRcd& rcd) {
  return create_impl<HcalDcsMap>(mInputs["DcsMap"]);
}

std::unique_ptr<HcalRecoParams> HcalTextCalibrations::produceRecoParams(const HcalRecoParamsRcd& rcd) {
  std::string const n = "RecoParams";
  return get_impl_topo<HcalRecoParams>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalLongRecoParams> HcalTextCalibrations::produceLongRecoParams(const HcalLongRecoParamsRcd& rcd) {
  std::string const n = "LongRecoParams";
  return get_impl_topo<HcalLongRecoParams>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalZDCLowGainFractions> HcalTextCalibrations::produceZDCLowGainFractions(
    const HcalZDCLowGainFractionsRcd& rcd) {
  std::string const n = "ZDCLowGainFractions";
  return get_impl_topo<HcalZDCLowGainFractions>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalTimingParams> HcalTextCalibrations::produceTimingParams(const HcalTimingParamsRcd& rcd) {
  std::string const n = "TimingParams";
  return get_impl_topo<HcalTimingParams>(mInputs[n], &rcd.get(mTokens[n]));
}
std::unique_ptr<HcalMCParams> HcalTextCalibrations::produceMCParams(const HcalMCParamsRcd& rcd) {
  std::string const n = "MCParams";
  return get_impl_topo<HcalMCParams>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalFlagHFDigiTimeParams> HcalTextCalibrations::produceFlagHFDigiTimeParams(
    const HcalFlagHFDigiTimeParamsRcd& rcd) {
  std::string const n = "FlagHFDigiTimeParams";
  return get_impl_topo<HcalFlagHFDigiTimeParams>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalSiPMParameters> HcalTextCalibrations::produceSiPMParameters(const HcalSiPMParametersRcd& rcd) {
  std::string const n = "SiPMParameters";
  return get_impl_topo<HcalSiPMParameters>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalSiPMCharacteristics> HcalTextCalibrations::produceSiPMCharacteristics(
    const HcalSiPMCharacteristicsRcd& rcd) {
  return create_impl<HcalSiPMCharacteristics>(mInputs["SiPMCharacteristics"]);
}

std::unique_ptr<HcalTPChannelParameters> HcalTextCalibrations::produceTPChannelParameters(
    const HcalTPChannelParametersRcd& rcd) {
  std::string const n = "TPChannelParameters";
  return get_impl_topo<HcalTPChannelParameters>(mInputs[n], &rcd.get(mTokens[n]));
}

std::unique_ptr<HcalTPParameters> HcalTextCalibrations::produceTPParameters(const HcalTPParametersRcd& rcd) {
  return get_impl<HcalTPParameters>(mInputs["TPParameters"]);
}
