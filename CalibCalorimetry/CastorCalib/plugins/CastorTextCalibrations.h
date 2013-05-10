#include <map>
#include <string>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ParameterSet;

class CastorPedestals;
class CastorPedestalWidths;
//class CastorGains;
//class CastorGainWidths;
class CastorQIEData;
//class CastorChannelQuality;
class CastorElectronicsMap;

class CastorPedestalsRcd;
class CastorPedestalWidthsRcd;
class CastorGainsRcd;
class CastorGainWidthsRcd;
class CastorQIEDataRcd;
class CastorChannelQualityRcd;
class CastorElectronicsMapRcd;
class CastorRecoParamsRcd;
class CastorSaturationCorrsRcd;

class CastorTextCalibrations : public edm::ESProducer,
		       public edm::EventSetupRecordIntervalFinder
{
public:
  CastorTextCalibrations (const edm::ParameterSet& );
  ~CastorTextCalibrations ();

  void produce () {};
  
protected:
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
			      const edm::IOVSyncValue& , 
			      edm::ValidityInterval&) ;

  std::auto_ptr<CastorPedestals> producePedestals (const CastorPedestalsRcd& rcd);
  std::auto_ptr<CastorPedestalWidths> producePedestalWidths (const CastorPedestalWidthsRcd& rcd);
  std::auto_ptr<CastorGains> produceGains (const CastorGainsRcd& rcd);
  std::auto_ptr<CastorGainWidths> produceGainWidths (const CastorGainWidthsRcd& rcd);
  std::auto_ptr<CastorQIEData> produceQIEData (const CastorQIEDataRcd& rcd);
  std::auto_ptr<CastorChannelQuality> produceChannelQuality (const CastorChannelQualityRcd& rcd);
  std::auto_ptr<CastorElectronicsMap> produceElectronicsMap (const CastorElectronicsMapRcd& rcd);
  std::auto_ptr<CastorRecoParams> produceRecoParams (const CastorRecoParamsRcd& rcd);
  std::auto_ptr<CastorSaturationCorrs> produceSaturationCorrs (const CastorSaturationCorrsRcd& rcd);

 private:
  std::map <std::string, std::string> mInputs;
};

