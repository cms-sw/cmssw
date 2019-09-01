#include "HcalTrigTowerGeometryESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include <memory>

HcalTrigTowerGeometryESProducer::HcalTrigTowerGeometryESProducer(const edm::ParameterSet& config)
    : topologyToken_{setWhatProduced(this).consumesFrom<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag{})} {}

HcalTrigTowerGeometryESProducer::~HcalTrigTowerGeometryESProducer(void) {}

std::unique_ptr<HcalTrigTowerGeometry> HcalTrigTowerGeometryESProducer::produce(const CaloGeometryRecord& iRecord) {
  const auto& hcalTopology = iRecord.get(topologyToken_);
  return std::make_unique<HcalTrigTowerGeometry>(&hcalTopology);
}

void HcalTrigTowerGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("HcalTrigTowerGeometryESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(HcalTrigTowerGeometryESProducer);
