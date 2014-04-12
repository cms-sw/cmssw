#ifndef FastSimulation_ParticlePropagator_MagneticFieldMapESProducer_H
#define FastSimulation_ParticlePropagator_MagneticFieldMapESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include <boost/shared_ptr.hpp>
#include <string>

class  MagneticFieldMapESProducer: public edm::ESProducer{
 public:
  MagneticFieldMapESProducer(const edm::ParameterSet & p);
  virtual ~MagneticFieldMapESProducer(); 
  boost::shared_ptr<MagneticFieldMap> produce(const MagneticFieldMapRecord &);
 private:
  boost::shared_ptr<MagneticFieldMap> _map;
  std::string _label;
};


#endif




