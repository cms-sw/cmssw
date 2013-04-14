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

    explicit ThingProducer(edm::ParameterSet const& ps);

    virtual ~ThingProducer();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

    virtual void beginRun(edm::Run& r, edm::EventSetup const& c);

    virtual void endRun(edm::Run& r, edm::EventSetup const& c);

    virtual void beginLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const& c);

    virtual void endLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const& c);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    ThingAlgorithm alg_;
    bool noPut_;
  };
}
#endif
