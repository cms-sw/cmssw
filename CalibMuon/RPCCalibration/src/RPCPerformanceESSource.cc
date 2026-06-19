#include "CalibMuon/RPCCalibration/interface/RPCPerformanceESSource.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
RPCPerformanceESSource::RPCPerformanceESSource(const edm::ParameterSet &pset) {
  setWhatProduced(this);
  findingRecord<RPCStripNoisesRcd>();
}

// -----------------------------------------------------------------------------
//
unique_ptr<RPCStripNoises> RPCPerformanceESSource::produce(const RPCStripNoisesRcd &) {
  RPCStripNoises *noise = makeNoise();

  return unique_ptr<RPCStripNoises>(noise);
}

// -----------------------------------------------------------------------------
//
