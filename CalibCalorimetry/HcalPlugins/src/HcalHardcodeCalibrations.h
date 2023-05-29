//
// Original Author:  Fedor Ratnikov Oct 21, 2005
//
// ESSource to generate default HCAL calibration objects
//
#include <map>
#include <unordered_map>
#include <string>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CalibCalorimetry/HcalAlgos/interface/HBHERecalibration.h"
#include "CondFormats/DataRecord/interface/HcalTPParametersRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HFRecalibration.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "CondFormats/DataRecord/interface/HcalAllRcds.h"

class ParameterSet;

class HcalPedestalsRcd;
class HcalPedestalWidthsRcd;
class HcalGainsRcd;
class HcalGainWidthsRcd;
class HcalPFCutsRcd;
class HcalQIEDataRcd;
class HcalQIETypesRcd;
class HcalChannelQualityRcd;
class HcalElectronicsMapRcd;
class HcalRespCorrsRcd;
class HcalZSThresholdsRcd;
class HcalL1TriggerObjectsRcd;
class HcalTimeCorrsRcd;
class HcalLUTCorrsRcd;
class HcalPFCorrsRcd;
class HcalValidationCorrsRcd;
class HcalLutMetadataRcd;
class HcalDcsRcd;
class HcalDcsMapRcd;
class HcalRecoParamsRcd;
class HcalLongRecoParamsRcd;
class HcalZDCLowGainFractionsRcd;
class HcalMCParamsRcd;
class HcalFlagHFDigiTimeParamsRcd;
class HcalTimingParamsRcd;
class HcalFrontEndMapRcd;
class HcalSiPMParametersRcd;
class HcalSiPMCharacteristicsRcd;
class HcalTPChannelParametersRcd;
class HcalTPParaamersRcd;

class HcalHardcodeCalibrations : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  HcalHardcodeCalibrations(const edm::ParameterSet&);
  ~HcalHardcodeCalibrations() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

  std::unique_ptr<HcalPedestals> producePedestals_(const HcalPedestalsRcd& rcd,
                                                   const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord>&,
                                                   bool eff);
  std::unique_ptr<HcalPedestalWidths> producePedestalWidths_(
      const HcalPedestalWidthsRcd& rcd, const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord>&, bool eff);
  std::unique_ptr<HcalPedestals> producePedestals(const HcalPedestalsRcd& rcd);
  std::unique_ptr<HcalPedestalWidths> producePedestalWidths(const HcalPedestalWidthsRcd& rcd);
  std::unique_ptr<HcalPedestals> produceEffectivePedestals(const HcalPedestalsRcd& rcd);
  std::unique_ptr<HcalPedestalWidths> produceEffectivePedestalWidths(const HcalPedestalWidthsRcd& rcd);
  std::unique_ptr<HcalGains> produceGains(const HcalGainsRcd& rcd);
  std::unique_ptr<HcalGainWidths> produceGainWidths(const HcalGainWidthsRcd& rcd);
  std::unique_ptr<HcalPFCuts> producePFCuts(const HcalPFCutsRcd& rcd);
  std::unique_ptr<HcalQIEData> produceQIEData(const HcalQIEDataRcd& rcd);
  std::unique_ptr<HcalQIETypes> produceQIETypes(const HcalQIETypesRcd& rcd);
  std::unique_ptr<HcalChannelQuality> produceChannelQuality(const HcalChannelQualityRcd& rcd);
  std::unique_ptr<HcalElectronicsMap> produceElectronicsMap(const HcalElectronicsMapRcd& rcd);

  std::unique_ptr<HcalRespCorrs> produceRespCorrs(const HcalRespCorrsRcd& rcd);
  std::unique_ptr<HcalZSThresholds> produceZSThresholds(const HcalZSThresholdsRcd& rcd);
  std::unique_ptr<HcalL1TriggerObjects> produceL1TriggerObjects(const HcalL1TriggerObjectsRcd& rcd);
  std::unique_ptr<HcalTimeCorrs> produceTimeCorrs(const HcalTimeCorrsRcd& rcd);
  std::unique_ptr<HcalLUTCorrs> produceLUTCorrs(const HcalLUTCorrsRcd& rcd);
  std::unique_ptr<HcalPFCorrs> producePFCorrs(const HcalPFCorrsRcd& rcd);

  std::unique_ptr<HcalValidationCorrs> produceValidationCorrs(const HcalValidationCorrsRcd& rcd);
  std::unique_ptr<HcalLutMetadata> produceLutMetadata(const HcalLutMetadataRcd& rcd);
  std::unique_ptr<HcalDcsValues> produceDcsValues(const HcalDcsRcd& rcd);
  std::unique_ptr<HcalDcsMap> produceDcsMap(const HcalDcsMapRcd& rcd);

  std::unique_ptr<HcalRecoParams> produceRecoParams(const HcalRecoParamsRcd& rcd);
  std::unique_ptr<HcalTimingParams> produceTimingParams(const HcalTimingParamsRcd& rcd);
  std::unique_ptr<HcalLongRecoParams> produceLongRecoParams(const HcalLongRecoParamsRcd& rcd);
  std::unique_ptr<HcalZDCLowGainFractions> produceZDCLowGainFractions(const HcalZDCLowGainFractionsRcd& rcd);

  std::unique_ptr<HcalMCParams> produceMCParams(const HcalMCParamsRcd& rcd);
  std::unique_ptr<HcalFlagHFDigiTimeParams> produceFlagHFDigiTimeParams(const HcalFlagHFDigiTimeParamsRcd& rcd);

  std::unique_ptr<HcalFrontEndMap> produceFrontEndMap(const HcalFrontEndMapRcd& rcd);

  std::unique_ptr<HcalSiPMParameters> produceSiPMParameters(const HcalSiPMParametersRcd& rcd);
  std::unique_ptr<HcalSiPMCharacteristics> produceSiPMCharacteristics(const HcalSiPMCharacteristicsRcd& rcd);
  std::unique_ptr<HcalTPChannelParameters> produceTPChannelParameters(const HcalTPChannelParametersRcd& rcd);
  std::unique_ptr<HcalTPParameters> produceTPParameters(const HcalTPParametersRcd& rcd);

private:
  enum {
    kPedestals,
    kPedestalWidths,
    kEffectivePedestals,
    kEffectivePedestalWidths,
    kGains,
    kGainWidths,
    kPFCuts,
    kQIEData,
    kQIETypes,
    kChannelQuality,
    kRespCorrs,
    kLUTCorrs,
    kPFCorrs,
    kTimeCorrs,
    kZSThresholds,
    kL1TriggerObjects,
    kElectronicsMap,
    kValidationCorrs,
    kLutMetadata,
    kRecoParams,
    kTimingParams,
    kLongRecoParams,
    kZDCLowGainFractions,
    kMCParams,
    kFlagHFDigiTimeParams,
    kFrontEndMap,
    kSiPMParameters,
    kTPChannelParameters
  };
  HcalDbHardcode dbHardcode;
  double iLumi;
  std::unique_ptr<HBHERecalibration> hb_recalibration;
  std::unique_ptr<HBHERecalibration> he_recalibration;
  std::unique_ptr<HFRecalibration> hf_recalibration;
  std::unordered_map<int, edm::ESGetToken<HcalTopology, HcalRecNumberingRecord>> topoTokens_;
  edm::ESGetToken<HBHEDarkening, HBHEDarkeningRecord> heDarkeningToken_;
  edm::ESGetToken<HBHEDarkening, HBHEDarkeningRecord> hbDarkeningToken_;
  bool switchGainWidthsForTrigPrims;
  bool setHEdsegm;
  bool setHBdsegm;
  bool useLayer0Weight;
  bool useIeta18depth1;
  bool testHEPlan1;
};
