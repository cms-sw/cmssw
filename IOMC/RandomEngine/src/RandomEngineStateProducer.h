// -*- C++ -*-
//
// Package:    RandomEngine
// Class:      RandomEngineStateProducer
// 
/** \class RandomEngineStateProducer

 Description: Gets the state of the random number engines from
the related service and stores it in the event and luminosity block.

 Implementation:  This simply copies from the cache in the
service and puts the product in the Event and LuminosityBlock.
The cache is filled at the beginning of processing for each
event or lumi by a call from the InputSource or EventProcessor
to the service. This module gets called later.

\author W. David Dagenhart, created October 4, 2006
  (originally in FWCore/Services)
*/

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class RandomEngineStateProducer : public edm::global::EDProducer<edm::BeginLuminosityBlockProducer> {
  public:
    explicit RandomEngineStateProducer(edm::ParameterSet const& pset);
    ~RandomEngineStateProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    virtual void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override;
    virtual void produce(edm::StreamID iID, edm::Event& ev, edm::EventSetup const& es) const override;
};
