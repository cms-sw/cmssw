#ifndef Integration_ThingWithMergeProducer_h
#define Integration_ThingWithMergeProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include <string>
#include <vector>

namespace edmtest {
  class ThingWithMergeProducer : public edm::one::EDProducer<edm::EndRunProducer,
  edm::BeginRunProducer,
  edm::BeginLuminosityBlockProducer,
  edm::EndLuminosityBlockProducer,
  edm::one::WatchRuns,
  edm::one::WatchLuminosityBlocks> {
  public:

    explicit ThingWithMergeProducer(edm::ParameterSet const& ps);

    virtual ~ThingWithMergeProducer();

    void produce(edm::Event& e, edm::EventSetup const& c) override;

    void beginRun(edm::Run const& r, edm::EventSetup const& c) override;

    void endRun(edm::Run const& r, edm::EventSetup const& c) override;

    void beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& c) override;

    void endLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& c) override;


    void beginRunProduce(edm::Run& r, edm::EventSetup const& c) override;
    
    void endRunProduce(edm::Run& r, edm::EventSetup const& c) override;
    
    void beginLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) override;
    
    void endLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const& c) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  private:

    typedef std::vector<std::string>::const_iterator Iter;

    bool changeIsEqualValue_;
    std::vector<std::string> labelsToGet_;
    bool noPut_;
  };
}

#endif
