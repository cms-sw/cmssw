//
// Original Author:  Fedor Ratnikov Oct 21, 2005
//
//
#include <map>
#include <string>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
class ParameterSet;

class HcalPedestalsRcd;
class HcalPedestalWidthsRcd;
class HcalGainsRcd;
class HcalGainWidthsRcd;
class HcalQIEDataRcd;
class HcalQIETypesRcd;
class HcalChannelQualityRcd;
class HcalElectronicsMapRcd;
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
class HcalCholeskyMatricesRcd;
class HcalCovarianceMatricesRcd;
class HcalRecoParamsRcd;
class HcalLongRecoParamsRcd;
class HcalZDCLowGainFractionsRcd;
class HcalMCParamsRcd;
class HcalFlagHFDigiTimeParamsRcd;
class HcalTimingParamsRcd;

class HcalTextCalibrations : public edm::ESProducer,
		       public edm::EventSetupRecordIntervalFinder
{
public:
  HcalTextCalibrations (const edm::ParameterSet& );
  ~HcalTextCalibrations ();

  void produce () {};
  
protected:
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
			      const edm::IOVSyncValue& , 
			      edm::ValidityInterval&) ;

  std::unique_ptr<HcalPedestals> producePedestals (const HcalPedestalsRcd& rcd);
  std::unique_ptr<HcalPedestalWidths> producePedestalWidths (const HcalPedestalWidthsRcd& rcd);
  std::unique_ptr<HcalGains> produceGains (const HcalGainsRcd& rcd);
  std::unique_ptr<HcalGainWidths> produceGainWidths (const HcalGainWidthsRcd& rcd);
  std::unique_ptr<HcalQIEData> produceQIEData (const HcalQIEDataRcd& rcd);
  std::unique_ptr<HcalQIETypes> produceQIETypes (const HcalQIETypesRcd& rcd);
  std::unique_ptr<HcalChannelQuality> produceChannelQuality (const HcalChannelQualityRcd& rcd);
  std::unique_ptr<HcalElectronicsMap> produceElectronicsMap (const HcalElectronicsMapRcd& rcd);

  std::unique_ptr<HcalRespCorrs> produceRespCorrs (const HcalRespCorrsRcd& rcd);
  std::unique_ptr<HcalZSThresholds> produceZSThresholds (const HcalZSThresholdsRcd& rcd);
  std::unique_ptr<HcalL1TriggerObjects> produceL1TriggerObjects (const HcalL1TriggerObjectsRcd& rcd);
  std::unique_ptr<HcalTimeCorrs> produceTimeCorrs (const HcalTimeCorrsRcd& rcd);
  std::unique_ptr<HcalLUTCorrs> produceLUTCorrs (const HcalLUTCorrsRcd& rcd);
  std::unique_ptr<HcalPFCorrs> producePFCorrs (const HcalPFCorrsRcd& rcd);

  std::unique_ptr<HcalRecoParams> produceRecoParams (const HcalRecoParamsRcd& rcd);
  std::unique_ptr<HcalLongRecoParams> produceLongRecoParams (const HcalLongRecoParamsRcd& rcd);
  std::unique_ptr<HcalZDCLowGainFractions> produceZDCLowGainFractions (const HcalZDCLowGainFractionsRcd& rcd);
  std::unique_ptr<HcalMCParams> produceMCParams (const HcalMCParamsRcd& rcd);
  std::unique_ptr<HcalFlagHFDigiTimeParams> produceFlagHFDigiTimeParams (const HcalFlagHFDigiTimeParamsRcd& rcd);

  std::unique_ptr<HcalValidationCorrs> produceValidationCorrs (const HcalValidationCorrsRcd& rcd);
  std::unique_ptr<HcalLutMetadata> produceLutMetadata (const HcalLutMetadataRcd& rcd);
  std::unique_ptr<HcalDcsValues> produceDcsValues (HcalDcsRcd const & rcd);
  std::unique_ptr<HcalDcsMap> produceDcsMap (const HcalDcsMapRcd& rcd);
  std::unique_ptr<HcalCholeskyMatrices> produceCholeskyMatrices (const HcalCholeskyMatricesRcd& rcd);
  std::unique_ptr<HcalCovarianceMatrices> produceCovarianceMatrices (const HcalCovarianceMatricesRcd& rcd);
  
  std::unique_ptr<HcalTimingParams> produceTimingParams (const HcalTimingParamsRcd& rcd);
 private:
  std::map <std::string, std::string> mInputs;
};

