#include "RecoLocalFastTime/Records/interface/MTDTimeCalibRecord.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <memory>

using namespace edm;

class MTDTimeCalibESProducer : public edm::ESProducer {
public:
  MTDTimeCalibESProducer(const edm::ParameterSet& p);
  ~MTDTimeCalibESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<MTDTimeCalib> produce(const MTDTimeCalibRecord&);

private:
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> ddToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> topoToken_;
  edm::ParameterSet pset_;
};

MTDTimeCalibESProducer::MTDTimeCalibESProducer(const edm::ParameterSet& p) {
  pset_ = p;
  setWhatProduced(this, "MTDTimeCalib").setConsumes(ddToken_).setConsumes(topoToken_);
}

// Configuration descriptions
void MTDTimeCalibESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("BTLTimeOffset", 0.)->setComment("Time offset (additive) to all the BTL RecHits [ns]");
  desc.add<double>("ETLTimeOffset", 0.)->setComment("Time offset (additive) to all the ETL RecHits [ns]");
  desc.add<double>("BTLLightCollTime", 0.2)->setComment("Light collection time for BTL tile geometry [ns]");
  desc.add<double>("BTLLightCollSlope", 0.075)
      ->setComment("Light collection slope for bar for BTL bar tile geometry [ns/cm]");
  descriptions.add("MTDTimeCalibESProducer", desc);
}

std::unique_ptr<MTDTimeCalib> MTDTimeCalibESProducer::produce(const MTDTimeCalibRecord& iRecord) {
  return std::make_unique<MTDTimeCalib>(pset_, &iRecord.get(ddToken_), &iRecord.get(topoToken_));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_EVENTSETUP_MODULE(MTDTimeCalibESProducer);
