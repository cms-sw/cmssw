#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripPedestalsESSource::SiStripPedestalsESSource( const edm::ParameterSet& pset ) {
  setWhatProduced( this );
  findingRecord<SiStripPedestalsRcd>();
}

// -----------------------------------------------------------------------------
//
std::auto_ptr<SiStripPedestals> SiStripPedestalsESSource::produce( const SiStripPedestalsRcd& ) { 
  
  SiStripPedestals* peds = makePedestals();
  
  if ( !peds ) {
    edm::LogWarning(mlESSources_)
      << "[SiStripPedestalsESSource::" << __func__ << "]"
      << " Null pointer to SiStripPedestals object!";
  }
  
  std::auto_ptr<SiStripPedestals> ptr(peds);
  return ptr;

}

// -----------------------------------------------------------------------------
//
void SiStripPedestalsESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {

  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
  
}
