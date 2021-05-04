#include "CalibTracker/SiStripESProducers/interface/SiStripGainESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripGainESSource::SiStripGainESSource(const edm::ParameterSet& pset) {
  setWhatProduced(this);
  findingRecord<SiStripApvGainRcd>();
}

// -----------------------------------------------------------------------------
//
std::unique_ptr<SiStripApvGain> SiStripGainESSource::produce(const SiStripApvGainRcd&) {
  SiStripApvGain* gain = makeGain();

  if (!gain) {
    edm::LogWarning(mlESSources_) << "[SiStripGainESSource::" << __func__ << "]"
                                  << " Null pointer to SiStripApvGain object!";
  }

  std::unique_ptr<SiStripApvGain> ptr(gain);
  return ptr;
}

// -----------------------------------------------------------------------------
//
void SiStripGainESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                         const edm::IOVSyncValue& iosv,
                                         edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}
