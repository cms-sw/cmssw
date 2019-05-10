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

class CastorHardcodeCalibrations : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CastorHardcodeCalibrations(const edm::ParameterSet&);
  ~CastorHardcodeCalibrations() override;

  void produce(){};

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

  std::unique_ptr<CastorPedestals> producePedestals(const CastorPedestalsRcd& rcd);
  std::unique_ptr<CastorPedestalWidths> producePedestalWidths(const CastorPedestalWidthsRcd& rcd);
  std::unique_ptr<CastorGains> produceGains(const CastorGainsRcd& rcd);
  std::unique_ptr<CastorGainWidths> produceGainWidths(const CastorGainWidthsRcd& rcd);
  std::unique_ptr<CastorQIEData> produceQIEData(const CastorQIEDataRcd& rcd);
  std::unique_ptr<CastorChannelQuality> produceChannelQuality(const CastorChannelQualityRcd& rcd);
  std::unique_ptr<CastorElectronicsMap> produceElectronicsMap(const CastorElectronicsMapRcd& rcd);
  std::unique_ptr<CastorRecoParams> produceRecoParams(const CastorRecoParamsRcd& rcd);
  std::unique_ptr<CastorSaturationCorrs> produceSaturationCorrs(const CastorSaturationCorrsRcd& rcd);
  bool h2mode_;
};
