#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTGPU.h"
#include "CalibTracker/Records/interface/SiPixelGainCalibrationForHLTGPURcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include <memory>

class SiPixelGainCalibrationForHLTGPUESProducer: public edm::ESProducer {
public:
  explicit SiPixelGainCalibrationForHLTGPUESProducer(const edm::ParameterSet& iConfig);
  std::unique_ptr<SiPixelGainCalibrationForHLTGPU> produce(const SiPixelGainCalibrationForHLTGPURcd& iRecord);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:
};

SiPixelGainCalibrationForHLTGPUESProducer::SiPixelGainCalibrationForHLTGPUESProducer(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);
}

void SiPixelGainCalibrationForHLTGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("siPixelGainCalibrationForHLTGPU", desc);
}

std::unique_ptr<SiPixelGainCalibrationForHLTGPU> SiPixelGainCalibrationForHLTGPUESProducer::produce(const SiPixelGainCalibrationForHLTGPURcd& iRecord) {
  edm::ESHandle<SiPixelGainCalibrationForHLT> gains;
  iRecord.getRecord<SiPixelGainCalibrationForHLTRcd>().get(gains);

  edm::ESHandle<TrackerGeometry> geom;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(geom);

  return std::make_unique<SiPixelGainCalibrationForHLTGPU>(*gains, *geom);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelGainCalibrationForHLTGPUESProducer);
