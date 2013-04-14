#ifndef Integration_ThingWithMergeProducer_h
#define Integration_ThingWithMergeProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include <string>
#include <vector>

namespace edmtest {
  class ThingWithMergeProducer : public edm::EDProducer {
  public:

    explicit ThingWithMergeProducer(edm::ParameterSet const& ps);

    virtual ~ThingWithMergeProducer();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

    virtual void beginRun(edm::Run& r, edm::EventSetup const& c);

    virtual void endRun(edm::Run& r, edm::EventSetup const& c);

    virtual void beginLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const& c);

    virtual void endLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const& c);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  private:

    typedef std::vector<std::string>::const_iterator Iter;

    bool changeIsEqualValue_;
    std::vector<std::string> labelsToGet_;
    bool noPut_;
  };
}

#endif
