#include "CalibTracker/SiStripESProducers/interface/SiStripHashedDetIdESProducer.h"
#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdESProducer::SiStripHashedDetIdESProducer( const edm::ParameterSet& pset ) {
  setWhatProduced( this, &SiStripHashedDetIdESProducer::produce );
}

// -----------------------------------------------------------------------------
//
SiStripHashedDetIdESProducer::~SiStripHashedDetIdESProducer() {}

// -----------------------------------------------------------------------------
//
std::auto_ptr<SiStripHashedDetId> SiStripHashedDetIdESProducer::produce( const SiStripHashedDetIdRcd& rcd ) { 
  
  SiStripHashedDetId* temp = make( rcd );
  
  if ( !temp ) {
    edm::LogWarning(mlCabling_)
      << "[SiStripHashedDetIdESProducer::" << __func__ << "]"
      << " Null pointer to SiStripHashedDetId object!";
  }
  
  std::auto_ptr<SiStripHashedDetId> ptr( temp );
  return ptr;
  
}

