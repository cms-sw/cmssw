<<<<<<< SiStripNoiseESSource.cc
// Last commit: $Id: SiStripNoiseESSource.cc,v 1.1 2008/05/14 10:00:02 giordano Exp $SiStripNoiseESSource
// Latest tag:  $Name: V03-00-04 $
=======
// Last commit: $Id: SiStripNoiseESSource.cc,v 1.2 2009/02/17 16:14:34 muzaffar Exp $SiStripNoiseESSource
// Latest tag:  $Name:  $
>>>>>>> 1.2
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripESProducers/src/SiStripNoiseESSource.cc,v $

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
