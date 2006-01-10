//
// Original Author:  Fedor Ratnikov Oct 21, 2005
// $Id: HcalTextCalibrations.h,v 1.1 2005/12/05 00:25:31 fedor Exp $
//
//
#include <map>
#include <string>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ParameterSet;

class HcalPedestals;
class HcalPedestalWidths;
class HcalGains;
class HcalGainWidths;
class HcalQIEData;
class HcalChannelQuality;
class HcalElectronicsMap;

class HcalPedestalsRcd;
class HcalPedestalWidthsRcd;
class HcalGainsRcd;
class HcalGainWidthsRcd;
class HcalQIEDataRcd;
class HcalChannelQualityRcd;
class HcalElectronicsMapRcd;

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

 private:
  std::map <std::string, std::string> mInputs;
};

