//
// Original Author:  Filippo Ambroglini
//         Created:  Fri Sep 29 17:10:41 CEST 2006
// $Id: MinimumBiasFilter.cc,v 1.3 2009/05/25 13:53:23 fabiocos Exp $
//
//
// system include files

#include "GeneratorInterface/GenFilters/interface/MinimumBiasFilter.h"

// user include files


#include "CLHEP/Random/RandFlat.h"

MinimumBiasFilter::MinimumBiasFilter(const edm::ParameterSet& iConfig):
  theEventFraction(0)
{
  theEventFraction=iConfig.getUntrackedParameter<double>("EventFraction");
}
  
  // ------------ method called on each new Event  ------------
bool MinimumBiasFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  /**
   * Wainting the real trigger for the 
   * MB we have developped 
   * a random one 
  */

  float rnd = CLHEP::RandFlat::shoot(0.,1.);
  
  if(rnd<=theEventFraction)
    return true;
  
  return false;
}
