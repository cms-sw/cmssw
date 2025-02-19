#include "CalibMuon/RPCCalibration/interface/RPCPerformanceESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
RPCPerformanceESSource::RPCPerformanceESSource( const edm::ParameterSet& pset ) {

  setWhatProduced( this );
  findingRecord<RPCStripNoisesRcd>();
}

// -----------------------------------------------------------------------------
//
auto_ptr<RPCStripNoises> RPCPerformanceESSource::produce( const RPCStripNoisesRcd&) { 
    
  RPCStripNoises* noise = makeNoise();
  
  auto_ptr<RPCStripNoises> ptr(noise);
  return ptr;

}

// -----------------------------------------------------------------------------
//
void RPCPerformanceESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
					   const edm::IOVSyncValue& iosv, 
					   edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
}


