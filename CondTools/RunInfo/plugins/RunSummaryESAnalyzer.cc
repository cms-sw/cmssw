#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;

namespace edmtest {
  class RunSummaryESAnalyzer : public edm::one::EDAnalyzer<> {
  private:
    const edm::ESGetToken<RunSummary, RunSummaryRcd> m_RunSummaryToken;

  public:
    explicit RunSummaryESAnalyzer(edm::ParameterSet const& p) : m_RunSummaryToken(esConsumes()) {
      edm::LogPrint("RunSummaryESAnalyzer") << "RunSummaryESAnalyzer" << std::endl;
    }
    explicit RunSummaryESAnalyzer(int i) {
      edm::LogPrint("RunSummaryESAnalyzer") << "RunSummaryESAnalyzer " << i << std::endl;
    }
    ~RunSummaryESAnalyzer() override { edm::LogPrint("RunSummaryESAnalyzer") << "~RunSummaryESAnalyzer " << std::endl; }
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void RunSummaryESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    edm::LogPrint("RunSummaryESAnalyzer") << "###RunSummaryESAnalyzer::analyze" << std::endl;

    // Context is not used.
    edm::LogPrint("RunSummaryESAnalyzer") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    edm::LogPrint("RunSummaryESAnalyzer") << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunSummaryRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogPrint("RunSummaryESAnalyzer") << "Record \"RunSummaryRcd"
                                            << "\" does not exist " << std::endl;
    }
    edm::LogPrint("RunSummaryESAnalyzer") << "got eshandle" << std::endl;
    edm::ESHandle<RunSummary> sum = context.getHandle(m_RunSummaryToken);
    edm::LogPrint("RunSummaryESAnalyzer") << "got context" << std::endl;
    const RunSummary* summary = sum.product();
    edm::LogPrint("RunSummaryESAnalyzer") << "got RunSummary* " << std::endl;

    edm::LogPrint("RunSummaryESAnalyzer") << "print  result" << std::endl;
    summary->printAllValues();
    std::vector<std::string> subdet = summary->getSubdtIn();
    edm::LogPrint("RunSummaryESAnalyzer") << "subdetector in the run " << std::endl;
    for (size_t i = 0; i < subdet.size(); i++) {
      edm::LogPrint("RunSummaryESAnalyzer") << "--> " << subdet[i] << std::endl;
    }
  }
  DEFINE_FWK_MODULE(RunSummaryESAnalyzer);
}  // namespace edmtest
