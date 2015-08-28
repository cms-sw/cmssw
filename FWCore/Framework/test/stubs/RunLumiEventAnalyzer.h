#ifndef Integration_RunLumiEventAnalyzer_h
#define Integration_RunLumiEventAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <vector>

namespace edm {
  class TriggerResults;
}

namespace edmtest {

  class RunLumiEventAnalyzer : public edm::EDAnalyzer {
  public:

    explicit RunLumiEventAnalyzer(edm::ParameterSet const& pset);

    virtual ~RunLumiEventAnalyzer() {}

    virtual void analyze(edm::Event const& event, edm::EventSetup const& es);
    virtual void beginRun(edm::Run const& run, edm::EventSetup const& es);
    virtual void endRun(edm::Run const& run, edm::EventSetup const& es);
    virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es);
    virtual void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es);
    virtual void endJob();
    virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

  private:

    std::vector<unsigned long long> expectedRunLumisEvents0_;
    std::vector<unsigned long long> expectedRunLumisEvents1_;
    std::vector<unsigned long long> *expectedRunLumisEvents_;
    int index_;
    bool verbose_;
    bool dumpTriggerResults_;
    int expectedEndingIndex0_;
    int expectedEndingIndex1_;
    int expectedEndingIndex_;
    edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  };
}

#endif
