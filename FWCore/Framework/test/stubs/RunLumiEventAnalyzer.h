#ifndef Integration_RunLumiEventAnalyzer_h
#define Integration_RunLumiEventAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <vector>

namespace edm {
  class TriggerResults;
}

namespace edmtest {

  class RunLumiEventAnalyzer : public edm::EDAnalyzer {
  public:

    explicit RunLumiEventAnalyzer(edm::ParameterSet const& pset);

    ~RunLumiEventAnalyzer() override {}

    void analyze(edm::Event const& event, edm::EventSetup const& es) override;
    void beginRun(edm::Run const& run, edm::EventSetup const& es) override;
    void endRun(edm::Run const& run, edm::EventSetup const& es) override;
    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) override;
    void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) override;
    void endJob() override;

  private:

    std::vector<unsigned long long> expectedRunLumisEvents0_;
    std::vector<unsigned long long> expectedRunLumisEvents1_;
    edm::propagate_const<std::vector<unsigned long long>*> expectedRunLumisEvents_;
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
