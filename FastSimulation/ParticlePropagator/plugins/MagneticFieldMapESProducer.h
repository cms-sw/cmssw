#ifndef FastSimulation_ParticlePropagator_MagneticFieldMapESProducer_H
#define FastSimulation_ParticlePropagator_MagneticFieldMapESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <string>

class MagneticFieldMapESProducer : public edm::ESProducer {
public:
  MagneticFieldMapESProducer(const edm::ParameterSet &p);
  ~MagneticFieldMapESProducer() override = default;
  std::unique_ptr<MagneticFieldMap> produce(const MagneticFieldMapRecord &);

private:
  const std::string label_;
  edm::ESGetToken<TrackerInteractionGeometry, TrackerInteractionGeometryRecord> tokenGeom_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenBField_;
};

#endif
