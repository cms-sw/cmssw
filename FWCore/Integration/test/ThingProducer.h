#ifndef EDMREFTEST_THINGPRODUCER_H
#define EDMREFTEST_THINGPRODUCER_H

/** \class ThingProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/CoreFramework/interface/CoreFrameworkFwd.h"
#include "FWCore/CoreFramework/interface/EDProducer.h"
#include "FWCore/FWCoreIntegration/test/ThingAlgorithm.h"

namespace edm {
  class ParameterSet;
}

namespace edmreftest {
  class ThingProducer : public edm::EDProducer {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit ThingProducer(edm::ParameterSet const& ps);

    virtual ~ThingProducer();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    ThingAlgorithm alg_;
  };
}
#endif
