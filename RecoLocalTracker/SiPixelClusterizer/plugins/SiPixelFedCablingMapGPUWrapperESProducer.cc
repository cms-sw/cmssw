#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPUWrapper.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h" // TODO: eventually use something more limited

#include <memory>

class SiPixelFedCablingMapGPUWrapperESProducer: public edm::ESProducer {
public:
  explicit SiPixelFedCablingMapGPUWrapperESProducer(const edm::ParameterSet& iConfig);
  std::unique_ptr<SiPixelFedCablingMapGPUWrapper> produce(const CkfComponentsRecord& iRecord);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::string cablingMapLabel_;
  bool useQuality_;
};

SiPixelFedCablingMapGPUWrapperESProducer::SiPixelFedCablingMapGPUWrapperESProducer(const edm::ParameterSet& iConfig):
  cablingMapLabel_(iConfig.getParameter<std::string>("CablingMapLabel")),
  useQuality_(iConfig.getParameter<bool>("UseQualityInfo"))
{
  std::string myname = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, myname);
}

void SiPixelFedCablingMapGPUWrapperESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  std::string label = "siPixelFedCablingMapGPUWrapper";
  desc.add<std::string>("ComponentName", "");
  desc.add<std::string>("CablingMapLabel","")->setComment("CablingMap label");
  desc.add<bool>("UseQualityInfo",false);

  descriptions.add(label, desc);
}

std::unique_ptr<SiPixelFedCablingMapGPUWrapper> SiPixelFedCablingMapGPUWrapperESProducer::produce(const CkfComponentsRecord& iRecord) {
  edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
  iRecord.getRecord<SiPixelFedCablingMapRcd>().get( cablingMapLabel_, cablingMap );

  const SiPixelQuality *quality = nullptr;
  if(useQuality_) {
    edm::ESTransientHandle<SiPixelQuality> qualityInfo;
    iRecord.getRecord<SiPixelQualityRcd>().get(qualityInfo);
    quality = qualityInfo.product();
  }

  edm::ESTransientHandle<TrackerGeometry> geom;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(geom);

  return std::make_unique<SiPixelFedCablingMapGPUWrapper>(*cablingMap, *geom, quality);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelFedCablingMapGPUWrapperESProducer);
