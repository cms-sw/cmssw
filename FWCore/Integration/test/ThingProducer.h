#ifndef Integration_ThingProducer_h
#define Integration_ThingProducer_h

/** \class ThingProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Integration/test/ThingAlgorithm.h"

namespace edmtest {
  class ThingProducer : public edm::one::EDProducer<edm::BeginRunProducer,
  edm::EndRunProducer,
  edm::EndLuminosityBlockProducer,
  edm::BeginLuminosityBlockProducer> {
  public:

    explicit ThingProducer(edm::ParameterSet const& ps);

    virtual ~ThingProducer();

    void produce(edm::Event& e, edm::EventSetup const& c) override;

    void beginRunProduce(edm::Run& r, edm::EventSetup const& c) override;

    void endRunProduce(edm::Run& r, edm::EventSetup const& c) override;

    void beginLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) override;

    void endLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    ThingAlgorithm alg_;
    bool noPut_;
  };
}
#endif
