#include "CalibTracker/SiStripESProducers/plugins/SiStripFedCablingESProducer.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripFedCablingESProducer::SiStripFedCablingESProducer( const edm::ParameterSet& pset ) {
  setWhatProduced( this );
  findingRecord<SiStripFedCablingRcd>();
}

// -----------------------------------------------------------------------------
//
std::auto_ptr<SiStripFedCabling> SiStripFedCablingESProducer::produce( const SiStripFedCablingRcd& ) { 
  
  SiStripFedCabling* cabling = makeFedCabling();
  
  if ( !cabling ) {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingESProducer::" << __func__ << "]"
      << " Null pointer to SiStripFedCabling object!";
  }
  
  std::auto_ptr<SiStripFedCabling> ptr(cabling);
  return ptr;

}

// -----------------------------------------------------------------------------
//
void SiStripFedCablingESProducer::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						  const edm::IOVSyncValue& iosv, 
						  edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
}
