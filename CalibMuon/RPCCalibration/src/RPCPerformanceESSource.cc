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

  std::cout<<"Sono nel costruttore di RPCPerformanceESSource"<<std::endl;
  setWhatProduced( this );
  std::cout<<"Sono nel costruttore dopo set"<<std::endl;
  findingRecord<RPCStripNoisesRcd>();
  std::cout<<"Sono nel costruttore dopo finding"<<std::endl;


}

// -----------------------------------------------------------------------------
//
auto_ptr<RPCStripNoises> RPCPerformanceESSource::produce( const RPCStripNoisesRcd&) { 
    
  std::cout<<"Sono nel produce di RPCPerformanceESSource"<<std::endl;
  RPCStripNoises* noise = makeNoise();
  
  auto_ptr<RPCStripNoises> ptr(noise);
  return ptr;

}

// -----------------------------------------------------------------------------
//
void RPCPerformanceESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
					   const edm::IOVSyncValue& iosv, 
					   edm::ValidityInterval& oValidity ) {
    std::cout<<"Sono nel setIntervalFor  di RPCPerformanceESSource"<<std::endl;
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;

  std::cout<<"Sono DOPO setIntervalFor  di RPCPerformanceESSource"<<std::endl;
}


