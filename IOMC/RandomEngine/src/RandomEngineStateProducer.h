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
// $Id: RandomEngineStateProducer.h,v 1.1 2006/10/23 15:27:41 wdd Exp $
//
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"


class RandomEngineStateProducer : public edm::EDProducer {
  public:
    explicit RandomEngineStateProducer(const edm::ParameterSet&);
    ~RandomEngineStateProducer();

  private:
    virtual void beginJob() ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
};
