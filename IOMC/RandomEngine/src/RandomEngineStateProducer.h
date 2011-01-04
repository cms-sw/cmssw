// -*- C++ -*-
//
// Package:    RandomEngine
// Class:      RandomEngineStateProducer
// 
/** \class RandomEngineStateProducer

 Description: Gets the state of the random number engines from
the related service and stores it in the event.

 Implementation:  This simply copies from the cache in the
service and puts the product in the Event and LuminosityBlock.
The cache is filled at the beginning of processing for eac
event or lumi by a call from the InputSource to the service.
This module gets called later.

\author W. David Dagenhart, created October 4, 2006
  (originally in FWCore/Services)
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class RandomEngineStateProducer : public edm::EDProducer {
  public:
    explicit RandomEngineStateProducer(edm::ParameterSet const& pset);
    ~RandomEngineStateProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    virtual void beginLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const& es);
    virtual void produce(edm::Event& ev, edm::EventSetup const& es);
};
