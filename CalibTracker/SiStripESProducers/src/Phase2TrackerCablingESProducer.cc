#include "CalibTracker/SiStripESProducers/interface/Phase2TrackerCablingESProducer.h"
#include "CondFormats/DataRecord/interface/Phase2TrackerCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
Phase2TrackerCablingESProducer::Phase2TrackerCablingESProducer(const edm::ParameterSet& pset) {
  setWhatProduced(this, &Phase2TrackerCablingESProducer::produce);
}

// -----------------------------------------------------------------------------
//
Phase2TrackerCablingESProducer::~Phase2TrackerCablingESProducer() {}

// -----------------------------------------------------------------------------
//
std::unique_ptr<Phase2TrackerCabling> Phase2TrackerCablingESProducer::produce(const Phase2TrackerCablingRcd& rcd) {
  Phase2TrackerCabling* temp = make(rcd);

  if (!temp) {
    edm::LogWarning("Phase2TrackerCabling") << "[Phase2TrackerCablingESProducer::" << __func__ << "]"
                                            << " Null pointer to Phase2TrackerCabling object!";
  }

  std::unique_ptr<Phase2TrackerCabling> ptr(temp);
  return ptr;
}
