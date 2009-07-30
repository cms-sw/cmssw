//
// Original Author:  Fedor Ratnikov Oct 21, 2005
// $Id: HcalTextCalibrations.h,v 1.7 2009/05/20 15:54:23 rofierzy Exp $
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
class HcalChannelQualityRcd;
class HcalElectronicsMapRcd;
class HcalRespCorrsRcd;
class HcalTimeCorrsRcd;
class HcalLUTCorrsRcd;
class HcalPFCorrsRcd;
class HcalZSThresholdsRcd;
class HcalL1TriggerObjectsRcd;
class HcalValidationCorrsRcd;

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

  std::auto_ptr<HcalPedestals> producePedestals (const HcalPedestalsRcd& rcd);
  std::auto_ptr<HcalPedestalWidths> producePedestalWidths (const HcalPedestalWidthsRcd& rcd);
  std::auto_ptr<HcalGains> produceGains (const HcalGainsRcd& rcd);
  std::auto_ptr<HcalGainWidths> produceGainWidths (const HcalGainWidthsRcd& rcd);
  std::auto_ptr<HcalQIEData> produceQIEData (const HcalQIEDataRcd& rcd);
  std::auto_ptr<HcalChannelQuality> produceChannelQuality (const HcalChannelQualityRcd& rcd);
  std::auto_ptr<HcalElectronicsMap> produceElectronicsMap (const HcalElectronicsMapRcd& rcd);

  std::auto_ptr<HcalRespCorrs> produceRespCorrs (const HcalRespCorrsRcd& rcd);
  std::auto_ptr<HcalZSThresholds> produceZSThresholds (const HcalZSThresholdsRcd& rcd);
  std::auto_ptr<HcalL1TriggerObjects> produceL1TriggerObjects (const HcalL1TriggerObjectsRcd& rcd);
  std::auto_ptr<HcalTimeCorrs> produceTimeCorrs (const HcalTimeCorrsRcd& rcd);
  std::auto_ptr<HcalLUTCorrs> produceLUTCorrs (const HcalLUTCorrsRcd& rcd);
  std::auto_ptr<HcalPFCorrs> producePFCorrs (const HcalPFCorrsRcd& rcd);
  std::auto_ptr<HcalValidationCorrs> produceValidationCorrs (const HcalValidationCorrsRcd& rcd);
 private:
  std::map <std::string, std::string> mInputs;
};

