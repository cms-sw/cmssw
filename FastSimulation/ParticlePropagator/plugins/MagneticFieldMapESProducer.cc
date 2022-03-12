#include "FastSimulation/ParticlePropagator/plugins/MagneticFieldMapESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

MagneticFieldMapESProducer::MagneticFieldMapESProducer(const edm::ParameterSet& p)
    : _label(p.getUntrackedParameter<std::string>("trackerGeometryLabel", "")) {
  //    theTrackerMaterial = p.getParameter<edm::ParameterSet>("TrackerMaterial");

  auto cc = setWhatProduced(this);
  tokenGeom_ = cc.consumes(edm::ESInputTag("", _label));
  tokenBField_ = cc.consumes();

  setWhatProduced(this);
}

MagneticFieldMapESProducer::~MagneticFieldMapESProducer() {}

std::unique_ptr<MagneticFieldMap> MagneticFieldMapESProducer::produce(const MagneticFieldMapRecord& iRecord) {
  const edm::ESHandle<TrackerInteractionGeometry>& theInteractionGeometry =
      iRecord.getRecord<TrackerInteractionGeometryRecord>().getHandle(tokenGeom_);
  const edm::ESHandle<MagneticField>& theMagneticField =
      iRecord.getRecord<IdealMagneticFieldRecord>().getHandle(tokenBField_);

  return std::make_unique<MagneticFieldMap>(theMagneticField.product(), theInteractionGeometry.product());
}

DEFINE_FWK_EVENTSETUP_MODULE(MagneticFieldMapESProducer);
