#include "DQM/TrackerCommon/interface/DetectorStateFilter.h" 
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include <iostream>
 
using namespace std;
//
// -- Constructor
//
DetectorStateFilter::DetectorStateFilter( const edm::ParameterSet & pset ) {
   verbose_      = pset.getUntrackedParameter<bool>( "DebugOn", false );
   detectorType_ = pset.getUntrackedParameter<string>( "DetectorType", "sistrip");
   nEvents_         = 0;
   nSelectedEvents_ = 0;
   detectorOn_  = false;
}
//
// -- Destructor
//
DetectorStateFilter::~DetectorStateFilter() {
}
 
bool DetectorStateFilter::filter( edm::Event & evt, edm::EventSetup const& es) {
  
  nEvents_++;

  edm::Handle<DcsStatus> dcsStatus;
  evt.getByLabel("DcsStatus", dcsStatus);
  if (dcsStatus.isValid()) {
    if (detectorType_ == "pixel") {
      if (dcsStatus->ready(DcsStatus::BPIX) && 
	  dcsStatus->ready(DcsStatus::FPIX)) detectorOn_ = true;
      else detectorOn_ = false;
    } else if (detectorType_ == "sistrip") {  
      if (dcsStatus->ready(DcsStatus::TIBTID) &&
	  dcsStatus->ready(DcsStatus::TOB) &&   
	  dcsStatus->ready(DcsStatus::TECp) &&  
	  dcsStatus->ready(DcsStatus::TECm)) detectorOn_ = true;     
      else detectorOn_ = false;
    }
  }
  if (detectorOn_) {
    nSelectedEvents_++; 
  } 
  if ( verbose_ ) cout << " Total Events " << nEvents_ 
                       << " Selected Events " << nSelectedEvents_ 
                       << " Detector State " << detectorOn_<<  endl;
  return detectorOn_;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DetectorStateFilter);

 
