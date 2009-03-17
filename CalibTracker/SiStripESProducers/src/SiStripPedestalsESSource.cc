<<<<<<< SiStripPedestalsESSource.cc
// Last commit: $Id: SiStripPedestalsESSource.cc,v 1.1 2008/05/14 10:00:02 giordano Exp $
// Latest tag:  $Name: V03-00-04 $
=======
// Last commit: $Id: SiStripPedestalsESSource.cc,v 1.2 2009/02/17 16:14:34 muzaffar Exp $
// Latest tag:  $Name:  $
>>>>>>> 1.2
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripESProducers/src/SiStripPedestalsESSource.cc,v $

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
