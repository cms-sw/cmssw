// Last commit: $Id: SiStripPedestalsESSource.cc,v 1.1 2006/12/22 12:04:33 bainbrid Exp $
// Latest tag:  $Name: V07-00-02-00 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripPedestals/src/SiStripPedestalsESSource.cc,v $

#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include <iostream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripPedestalsESSource::SiStripPedestalsESSource( const edm::ParameterSet& pset ) {
  setWhatProduced( this );
  findingRecord<SiStripPedestalsRcd>();
}

// -----------------------------------------------------------------------------
//
auto_ptr<SiStripPedestals> SiStripPedestalsESSource::produce( const SiStripPedestalsRcd& ) { 
  
  SiStripPedestals* peds = makePedestals();
  
  if ( !peds ) {
    edm::LogWarning(mlESSources_)
      << "[SiStripPedestalsESSource::" << __func__ << "]"
      << " Null pointer to SiStripPedestals object!";
  }
  
  auto_ptr<SiStripPedestals> ptr(peds);
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
