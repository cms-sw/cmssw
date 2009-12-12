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

  edm::Handle<DcsStatusCollection> dcsStatus;
  evt.getByLabel("scalersRawToDigi", dcsStatus);
  if (dcsStatus.isValid()) {
    if (detectorType_ == "pixel") {
      if ((*dcsStatus)[0].ready(DcsStatus::BPIX) && 
	  (*dcsStatus)[0].ready(DcsStatus::FPIX)) {
	detectorOn_ = true;
	nSelectedEvents_++;
      } else detectorOn_ = false;
      if ( verbose_ ) cout << " Total Events " << nEvents_ 
			   << " Selected Events " << nSelectedEvents_ 
			   << " DCS States : " << " BPix " << (*dcsStatus)[0].ready(DcsStatus::BPIX) 
			   << " FPix " << (*dcsStatus)[0].ready(DcsStatus::FPIX)
			   << " Detector State " << detectorOn_<<  endl;           
    } else if (detectorType_ == "sistrip") {  
      if ((*dcsStatus)[0].ready(DcsStatus::TIBTID) &&
	  (*dcsStatus)[0].ready(DcsStatus::TOB) &&   
	  (*dcsStatus)[0].ready(DcsStatus::TECp) &&  
	  (*dcsStatus)[0].ready(DcsStatus::TECm)) {
        detectorOn_ = true;             
	nSelectedEvents_++;
      } else detectorOn_ = false;
      if ( verbose_ ) cout << " Total Events " << nEvents_ 
			   << " Selected Events " << nSelectedEvents_ 
			   << " DCS States : " << " TEC- " << (*dcsStatus)[0].ready(DcsStatus::TECm) 
			   << " TEC+ " << (*dcsStatus)[0].ready(DcsStatus::TECp)
			   << " TIB/TID " << (*dcsStatus)[0].ready(DcsStatus::TIBTID) 
			   << " TOB " << (*dcsStatus)[0].ready(DcsStatus::TOB)   
			   << " Detector States " << detectorOn_<<  endl;      
    }
  }
  return detectorOn_;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DetectorStateFilter);

 
