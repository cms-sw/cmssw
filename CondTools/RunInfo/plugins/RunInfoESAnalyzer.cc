#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;

namespace edmtest {
  class RunInfoESAnalyzer : public edm::one::EDAnalyzer<> {
  private:
    const edm::ESGetToken<RunInfo, RunInfoRcd> m_RunInfoToken;

  public:
    explicit RunInfoESAnalyzer(edm::ParameterSet const& p) : m_RunInfoToken(esConsumes()) {
      edm::LogPrint("RunInfoESAnalyzer") << "RunInfoESAnalyzer" << std::endl;
    }
    explicit RunInfoESAnalyzer(int i) { edm::LogPrint("RunInfoESAnalyzer") << "RunInfoESAnalyzer " << i << std::endl; }
    ~RunInfoESAnalyzer() override { edm::LogPrint("RunInfoESAnalyzer") << "~RunInfoESAnalyzer " << std::endl; }
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  };
  void RunInfoESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    edm::LogPrint("RunInfoESAnalyzer") << "###RunInfoESAnalyzer::analyze" << std::endl;

    // Context is not used.
    edm::LogPrint("RunInfoESAnalyzer") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    edm::LogPrint("RunInfoESAnalyzer") << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogPrint("RunInfoESAnalyzer") << "Record \"RunInfoRcd"
                                         << "\" does not exist " << std::endl;
    }
    edm::LogPrint("RunInfoESAnalyzer") << "got eshandle" << std::endl;
    edm::ESHandle<RunInfo> sum = context.getHandle(m_RunInfoToken);
    edm::LogPrint("RunInfoESAnalyzer") << "got context" << std::endl;
    const RunInfo* summary = sum.product();
    edm::LogPrint("RunInfoESAnalyzer") << "got RunInfo* " << std::endl;
    edm::LogPrint("RunInfoESAnalyzer") << "print  result" << std::endl;
    summary->printAllValues();
    /*
    std::vector<std::string> subdet = summary->getSubdtIn();
    edm::LogPrint("RunInfoESAnalyzer")<<"subdetector in the run "<< std::endl;
    for (size_t i=0; i<subdet.size(); i++){
      edm::LogPrint("RunInfoESAnalyzer")<<"--> " << subdet[i] << std::endl;
    }
    */
  }
  DEFINE_FWK_MODULE(RunInfoESAnalyzer);
}  // namespace edmtest
