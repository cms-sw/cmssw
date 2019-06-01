#ifndef FastSimulation_ParticlePropagator_MagneticFieldMapESProducer_H
#define FastSimulation_ParticlePropagator_MagneticFieldMapESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include <memory>
#include <string>

class MagneticFieldMapESProducer : public edm::ESProducer {
public:
  MagneticFieldMapESProducer(const edm::ParameterSet &p);
  ~MagneticFieldMapESProducer() override;
  std::unique_ptr<MagneticFieldMap> produce(const MagneticFieldMapRecord &);

private:
  std::string _label;
};

#endif
