#ifndef CalibTracker_SiStripESProducers_Phase2TrackerCablingESProducer_H
#define CalibTracker_SiStripESProducers_Phase2TrackerCablingESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class Phase2TrackerCabling;
class Phase2TrackerCablingRcd;

class Phase2TrackerCablingESProducer : public edm::ESProducer {
public:
  Phase2TrackerCablingESProducer(const edm::ParameterSet&);
  ~Phase2TrackerCablingESProducer() override;

  virtual std::unique_ptr<Phase2TrackerCabling> produce(const Phase2TrackerCablingRcd&);

private:
  Phase2TrackerCablingESProducer(const Phase2TrackerCablingESProducer&) = delete;
  const Phase2TrackerCablingESProducer& operator=(const Phase2TrackerCablingESProducer&) = delete;

  virtual Phase2TrackerCabling* make(const Phase2TrackerCablingRcd&) = 0;
};

#endif  // CalibTracker_SiStripESProducers_Phase2TrackerCablingESProducer_H
