#ifndef CalibTracker_SiStripESProducers_Phase2TrackerCablingCfgESSource_H
#define CalibTracker_SiStripESProducers_Phase2TrackerCablingCfgESSource_H

#include "CalibTracker/SiStripESProducers/interface/Phase2TrackerCablingESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Phase2TrackerCabling;
class Phase2TrackerCablingRcd;

class Phase2TrackerCablingCfgESSource : public Phase2TrackerCablingESProducer,
                                        public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit Phase2TrackerCablingCfgESSource(const edm::ParameterSet&);
  ~Phase2TrackerCablingCfgESSource() override;

private:
  // Builds cabling map based on the ParameterSet
  Phase2TrackerCabling* make(const Phase2TrackerCablingRcd&) override;

  // The configuration used to generated the cabling record
  edm::ParameterSet pset_;
};

#endif  // CalibTracker_SiStripESProducers_Phase2TrackerCablingCfgESSource_H
