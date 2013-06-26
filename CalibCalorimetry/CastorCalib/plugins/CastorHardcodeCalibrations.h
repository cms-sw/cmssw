// ESSource to generate default HCAL/CASTOR calibration objects 
//
#include <map>
#include <string>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CastorObjects/interface/AllObjects.h"
class ParameterSet;

class CastorPedestalsRcd;
class CastorPedestalWidthsRcd;
class CastorGainsRcd;
class CastorGainWidthsRcd;
class CastorQIEDataRcd;
class CastorChannelQualityRcd;
class CastorElectronicsMapRcd;
class CastorRecoParamsRcd;
class CastorSaturationCorrsRcd;

class CastorHardcodeCalibrations : public edm::ESProducer,
		       public edm::EventSetupRecordIntervalFinder
{
public:
  CastorHardcodeCalibrations (const edm::ParameterSet& );
  ~CastorHardcodeCalibrations ();

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
  bool h2mode_;
};

