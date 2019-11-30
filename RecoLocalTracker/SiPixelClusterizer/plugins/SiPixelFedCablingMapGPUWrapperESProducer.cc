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

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"  // TODO: eventually use something more limited

#include <memory>

class SiPixelFedCablingMapGPUWrapperESProducer : public edm::ESProducer {
public:
  explicit SiPixelFedCablingMapGPUWrapperESProducer(const edm::ParameterSet& iConfig);
  std::unique_ptr<SiPixelFedCablingMapGPUWrapper> produce(const CkfComponentsRecord& iRecord);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;
  edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> qualityToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
  bool useQuality_;
};

SiPixelFedCablingMapGPUWrapperESProducer::SiPixelFedCablingMapGPUWrapperESProducer(const edm::ParameterSet& iConfig)
    : useQuality_(iConfig.getParameter<bool>("UseQualityInfo")) {
  std::string component = iConfig.getParameter<std::string>("ComponentName");
  auto cc = setWhatProduced(this, component);
  cc.setConsumes(cablingMapToken_, edm::ESInputTag{"", iConfig.getParameter<std::string>("CablingMapLabel")});
  if (useQuality_) {
    cc.setConsumes(qualityToken_);
  }
  cc.setConsumes(geometryToken_);
}

void SiPixelFedCablingMapGPUWrapperESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  std::string label = "siPixelFedCablingMapGPUWrapper";
  desc.add<std::string>("ComponentName", "");
  desc.add<std::string>("CablingMapLabel", "")->setComment("CablingMap label");
  desc.add<bool>("UseQualityInfo", false);

  descriptions.add(label, desc);
}

std::unique_ptr<SiPixelFedCablingMapGPUWrapper> SiPixelFedCablingMapGPUWrapperESProducer::produce(
    const CkfComponentsRecord& iRecord) {
  auto cablingMap = iRecord.getTransientHandle(cablingMapToken_);

  const SiPixelQuality* quality = nullptr;
  if (useQuality_) {
    auto qualityInfo = iRecord.getTransientHandle(qualityToken_);
    quality = qualityInfo.product();
  }

  auto geom = iRecord.getTransientHandle(geometryToken_);

  return std::make_unique<SiPixelFedCablingMapGPUWrapper>(*cablingMap, *geom, quality);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelFedCablingMapGPUWrapperESProducer);
