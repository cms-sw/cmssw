// Last commit: $Id: SiStripNoiseESSource.cc,v 1.1 2006/12/22 12:04:33 bainbrid Exp $SiStripNoiseESSource
// Latest tag:  $Name: V07-00-02-00 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripPedestals/src/SiStripNoiseESSource.cc,v $

#include "CalibTracker/SiStripESProducers/interface/SiStripNoiseESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include <iostream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripNoiseESSource::SiStripNoiseESSource( const edm::ParameterSet& pset ) {
  setWhatProduced( this );
  findingRecord<SiStripNoisesRcd>();
}

// -----------------------------------------------------------------------------
//
auto_ptr<SiStripNoises> SiStripNoiseESSource::produce( const SiStripNoisesRcd& ) { 
  
  SiStripNoises* noise = makeNoise();
  
  if ( !noise ) {
    edm::LogWarning(mlESSources_)
      << "[SiStripNoiseESSource::" << __func__ << "]"
      << " Null pointer to SiStripNoises object!";
  }
  
  auto_ptr<SiStripNoises> ptr(noise);
  return ptr;

}

// -----------------------------------------------------------------------------
//
void SiStripNoiseESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
					   const edm::IOVSyncValue& iosv, 
					   edm::ValidityInterval& oValidity ) {
  
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
  
}
