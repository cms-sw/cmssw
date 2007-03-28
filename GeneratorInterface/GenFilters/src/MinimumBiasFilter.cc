//
// Original Author:  Filippo Ambroglini
//         Created:  Fri Sep 29 17:10:41 CEST 2006
// $Id: MinimumBiasFilter.cc,v 1.1 2006/10/06 08:10:24 fambrogl Exp $
//
//
// system include files
#include <memory>

#include "GeneratorInterface/GenFilters/interface/MinimumBiasFilter.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

  float rnd = RandFlat::shoot(0.,1.);
  
  if(rnd<=theEventFraction)
    return true;
  
  return false;
}
