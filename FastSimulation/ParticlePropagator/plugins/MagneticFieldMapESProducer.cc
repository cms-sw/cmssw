#include "FastSimulation/ParticlePropagator/plugins/MagneticFieldMapESProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

MagneticFieldMapESProducer::MagneticFieldMapESProducer(const edm::ParameterSet& p)
    : label_(p.getUntrackedParameter<std::string>("trackerGeometryLabel", "")) {
  auto cc = setWhatProduced(this);
  tokenGeom_ = cc.consumes(edm::ESInputTag("", label_));
  tokenBField_ = cc.consumes();
}

std::unique_ptr<MagneticFieldMap> MagneticFieldMapESProducer::produce(const MagneticFieldMapRecord& iRecord) {
  auto theInteractionGeometry = &(iRecord.getRecord<TrackerInteractionGeometryRecord>().get(tokenGeom_));
  auto theMagneticField = &(iRecord.getRecord<IdealMagneticFieldRecord>().get(tokenBField_));

  return std::make_unique<MagneticFieldMap>(theMagneticField, theInteractionGeometry);
}

DEFINE_FWK_EVENTSETUP_MODULE(MagneticFieldMapESProducer);
