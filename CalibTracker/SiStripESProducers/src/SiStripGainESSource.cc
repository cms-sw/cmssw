// Last commit: $Id: SiStripGainESSource.cc,v 1.1 2008/09/22 17:55:03 bainbrid Exp $
// Latest tag:  $Name: V03-00-00-00 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripESProducers/src/SiStripGainESSource.cc,v $

#include "CalibTracker/SiStripESProducers/interface/SiStripGainESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "boost/cstdint.hpp"
#include <iostream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripGainESSource::SiStripGainESSource( const edm::ParameterSet& pset ) {
  setWhatProduced( this );
  findingRecord<SiStripApvGainRcd>();
}

// -----------------------------------------------------------------------------
//
auto_ptr<SiStripApvGain> SiStripGainESSource::produce( const SiStripApvGainRcd& ) { 
  
  SiStripApvGain* gain = makeGain();
  
  if ( !gain ) {
    edm::LogWarning(mlESSources_)
      << "[SiStripGainESSource::" << __func__ << "]"
      << " Null pointer to SiStripApvGain object!";
  }
  
  auto_ptr<SiStripApvGain> ptr(gain);
  return ptr;

}

// -----------------------------------------------------------------------------
//
void SiStripGainESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
					  const edm::IOVSyncValue& iosv, 
					  edm::ValidityInterval& oValidity ) {
  
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
  
}
