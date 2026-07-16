#ifndef CalibTracker_SiStripESProducers_SiStripQualityFakeESSource
#define CalibTracker_SiStripESProducers_SiStripQualityFakeESSource

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

//
// class declaration
//

class SiStripQualityFakeESSource : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  SiStripQualityFakeESSource(const edm::ParameterSet&);
  ~SiStripQualityFakeESSource() override {}
  SiStripQualityFakeESSource(const SiStripQualityFakeESSource&) = delete;
  const SiStripQualityFakeESSource& operator=(const SiStripQualityFakeESSource&) = delete;

  std::unique_ptr<SiStripQuality> produce(const SiStripQualityRcd&);
};

#endif
