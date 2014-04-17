#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDCSStatus.h"

#include <iostream>
//
// -- Class Deleration
//
 
class SiStripDCSFilter : public edm::EDFilter {
 public:
 SiStripDCSFilter( const edm::ParameterSet & );
 ~SiStripDCSFilter();

  private:
  bool filter( edm::Event &, edm::EventSetup const& ) override;
  SiStripDCSStatus* dcsStatus_;  
};
 
//
// -- Constructor
//
SiStripDCSFilter::SiStripDCSFilter( const edm::ParameterSet & pset ) {
  dcsStatus_ = new SiStripDCSStatus(consumesCollector()); 
}
//
// -- Destructor
//
SiStripDCSFilter::~SiStripDCSFilter() {
  if (dcsStatus_) delete dcsStatus_;
}
 
bool SiStripDCSFilter::filter( edm::Event & evt, edm::EventSetup const& es) {

  return (dcsStatus_->getStatus(evt, es));   
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDCSFilter);

 
