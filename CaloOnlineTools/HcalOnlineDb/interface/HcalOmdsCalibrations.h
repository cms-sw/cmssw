//
// Original Author:  Gena Kukartsev Mar 11, 2009
// Adapted from HcalTextCalibrations
// $Id: HcalOmdsCalibrations.h,v 1.2 2009/03/24 14:33:27 kukartse Exp $
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
class HcalZSThresholdsRcd;
class HcalL1TriggerObjectsRcd;

class HcalOmdsCalibrations : public edm::ESProducer,
		       public edm::EventSetupRecordIntervalFinder
{
public:
  HcalOmdsCalibrations (const edm::ParameterSet& );
  ~HcalOmdsCalibrations ();

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
 private:
  std::map <std::string, std::string> mInputs;
  std::map <std::string, std::string> mVersion;
  std::map <std::string, int> mSubversion;
  std::map <std::string, int> mIOVBegin;
  std::map <std::string, std::string> mAccessor;
  std::map <std::string, std::string> mQuery;
};

