//
// Original Author:  Fedor Ratnikov Oct 21, 2005
//
//
#include <map>
#include <string>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/DataRecord/interface/HcalTPParametersRcd.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"
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
class HcalFrontEndMapRcd;
class HcalRespCorrsRcd;
class HcalTimeCorrsRcd;
class HcalLUTCorrsRcd;
class HcalPFCorrsRcd;
class HcalZSThresholdsRcd;
class HcalL1TriggerObjectsRcd;
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
class HcalSiPMParametersRcd;
class HcalSiPMCharacteristicsRcd;
class HcalTPChannelParametersRcd;
class HcalTPParaamersRcd;

class HcalTextCalibrations : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  HcalTextCalibrations(const edm::ParameterSet&);
  ~HcalTextCalibrations() override;

  void produce(){};

  template <class T>
  class CheckGetObject {
  public:
    CheckGetObject(const HcalTopology* topo) {}
    std::unique_ptr<T> operator()(std::istream& inStream) {
      auto result = makeResult();
      if (!HcalDbASCIIIO::getObject(inStream, &*result))
        result.reset(nullptr);
      return result;
    }
    virtual ~CheckGetObject() = default;

  protected:
    virtual std::unique_ptr<T> makeResult() { return std::make_unique<T>(); }
  };
  template <class T>
  class CheckGetObjectTopo : public CheckGetObject<T> {
  public:
    CheckGetObjectTopo(const HcalTopology* topo) : CheckGetObject<T>(topo), topo_(topo) {}
    ~CheckGetObjectTopo() override = default;

  protected:
    std::unique_ptr<T> makeResult() override { return std::make_unique<T>(topo_); }

  private:
    const HcalTopology* topo_;
  };
  template <class T>
  class CheckCreateObject {
  public:
    CheckCreateObject(const HcalTopology* topo) {}
    std::unique_ptr<T> operator()(std::istream& inStream) { return HcalDbASCIIIO::createObject<T>(inStream); }
  };

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

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
  std::unique_ptr<HcalFrontEndMap> produceFrontEndMap(const HcalFrontEndMapRcd& rcd);

  std::unique_ptr<HcalRespCorrs> produceRespCorrs(const HcalRespCorrsRcd& rcd);
  std::unique_ptr<HcalZSThresholds> produceZSThresholds(const HcalZSThresholdsRcd& rcd);
  std::unique_ptr<HcalL1TriggerObjects> produceL1TriggerObjects(const HcalL1TriggerObjectsRcd& rcd);
  std::unique_ptr<HcalTimeCorrs> produceTimeCorrs(const HcalTimeCorrsRcd& rcd);
  std::unique_ptr<HcalLUTCorrs> produceLUTCorrs(const HcalLUTCorrsRcd& rcd);
  std::unique_ptr<HcalPFCorrs> producePFCorrs(const HcalPFCorrsRcd& rcd);

  std::unique_ptr<HcalRecoParams> produceRecoParams(const HcalRecoParamsRcd& rcd);
  std::unique_ptr<HcalLongRecoParams> produceLongRecoParams(const HcalLongRecoParamsRcd& rcd);
  std::unique_ptr<HcalZDCLowGainFractions> produceZDCLowGainFractions(const HcalZDCLowGainFractionsRcd& rcd);
  std::unique_ptr<HcalMCParams> produceMCParams(const HcalMCParamsRcd& rcd);
  std::unique_ptr<HcalFlagHFDigiTimeParams> produceFlagHFDigiTimeParams(const HcalFlagHFDigiTimeParamsRcd& rcd);

  std::unique_ptr<HcalValidationCorrs> produceValidationCorrs(const HcalValidationCorrsRcd& rcd);
  std::unique_ptr<HcalLutMetadata> produceLutMetadata(const HcalLutMetadataRcd& rcd);
  std::unique_ptr<HcalDcsValues> produceDcsValues(HcalDcsRcd const& rcd);
  std::unique_ptr<HcalDcsMap> produceDcsMap(const HcalDcsMapRcd& rcd);

  std::unique_ptr<HcalTimingParams> produceTimingParams(const HcalTimingParamsRcd& rcd);
  std::unique_ptr<HcalSiPMParameters> produceSiPMParameters(const HcalSiPMParametersRcd& rcd);
  std::unique_ptr<HcalSiPMCharacteristics> produceSiPMCharacteristics(const HcalSiPMCharacteristicsRcd& rcd);
  std::unique_ptr<HcalTPChannelParameters> produceTPChannelParameters(const HcalTPChannelParametersRcd& rcd);
  std::unique_ptr<HcalTPParameters> produceTPParameters(const HcalTPParametersRcd& rcd);

private:
  std::map<std::string, std::string> mInputs;
  std::unordered_map<std::string, edm::ESGetToken<HcalTopology, HcalRecNumberingRecord>> mTokens;
};
