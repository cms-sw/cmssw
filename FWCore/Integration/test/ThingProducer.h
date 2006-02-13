#ifndef Integration_ThingProducer_h
#define Integration_ThingProducer_h

/** \class ThingProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Integration/test/ThingAlgorithm.h"

namespace edmtest {
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
