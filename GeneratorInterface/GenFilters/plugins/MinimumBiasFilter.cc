//
// Original Author:  Filippo Ambroglini
//         Created:  Fri Sep 29 17:10:41 CEST 2006
//
//
// system include files

#include "GeneratorInterface/GenFilters/plugins/MinimumBiasFilter.h"

// user include files

#include "CLHEP/Random/RandFlat.h"

MinimumBiasFilter::MinimumBiasFilter(const edm::ParameterSet& iConfig) : theEventFraction(0) {
  theEventFraction = iConfig.getUntrackedParameter<double>("EventFraction");
}

// ------------ method called on each new Event  ------------
bool MinimumBiasFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  /**
   * Wainting the real trigger for the 
   * MB we have developped 
   * a random one 
  */

  float rnd = CLHEP::RandFlat::shoot(0., 1.);

  if (rnd <= theEventFraction)
    return true;

  return false;
}
