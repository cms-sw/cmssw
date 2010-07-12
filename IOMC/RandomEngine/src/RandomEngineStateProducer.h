// -*- C++ -*-
//
// Package:    RandomEngine
// Class:      RandomEngineStateProducer
// 
/**\class RandomEngineStateProducer RandomEngineStateProducer.h IOMC/RandomEngine/src/RandomEngineStateProducer.h

 Description: Gets the state of the random number engines from
the related service and stores it in the event.

 Implementation:  This simply copies from the cache in the
service, does a small amount of formatting, and puts the object
in the event.  The cache is filled at the beginning of processing
for each event by a call from the InputSource to the service.
This module gets called later.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  Wed Oct  4 09:38:47 CDT 2006
//
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {
  class ConfigurationDescriptions;
}

class RandomEngineStateProducer : public edm::EDProducer {
  public:
    explicit RandomEngineStateProducer(edm::ParameterSet const&);
    ~RandomEngineStateProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    virtual void beginJob();
    virtual void produce(edm::Event&, edm::EventSetup const&);
    virtual void endJob();
};
