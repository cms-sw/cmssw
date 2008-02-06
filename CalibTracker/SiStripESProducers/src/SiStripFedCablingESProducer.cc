#include "CalibTracker/SiStripESProducers/interface/SiStripFedCablingESProducer.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripFedCablingESProducer::SiStripFedCablingESProducer( const edm::ParameterSet& pset ) {
  setWhatProduced( this, &SiStripFedCablingESProducer::produce );
}

// -----------------------------------------------------------------------------
//
SiStripFedCablingESProducer::~SiStripFedCablingESProducer() {}

// -----------------------------------------------------------------------------
//
std::auto_ptr<SiStripFedCabling> SiStripFedCablingESProducer::produce( const SiStripFedCablingRcd& rcd ) { 
  
  SiStripFedCabling* temp = make( rcd );
  
  if ( !temp ) {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingESProducer::" << __func__ << "]"
      << " Null pointer to SiStripFedCabling object!";
  }
  
  std::auto_ptr<SiStripFedCabling> ptr( temp );
  return ptr;

}
