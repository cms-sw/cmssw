#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

namespace edmtest {
  class LHCInfoESAnalyzer : public edm::one::EDAnalyzer<> {
  private:
    const edm::ESGetToken<LHCInfo, LHCInfoRcd> m_LHCInfoToken;

  public:
    explicit LHCInfoESAnalyzer(edm::ParameterSet const& p) : m_LHCInfoToken(esConsumes()) {
      edm::LogPrint("LHCInfoESAnalyzer") << "LHCInfoESAnalyzer" << std::endl;
    }
    explicit LHCInfoESAnalyzer(int i) { edm::LogPrint("LHCInfoESAnalyzer") << "LHCInfoESAnalyzer " << i << std::endl; }
    ~LHCInfoESAnalyzer() override { edm::LogPrint("LHCInfoESAnalyzer") << "~LHCInfoESAnalyzer " << std::endl; }
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  };

  void LHCInfoESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    edm::LogPrint("LHCInfoESAnalyzer") << "###LHCInfoESAnalyzer::analyze" << std::endl;
    edm::LogPrint("LHCInfoESAnalyzer") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    edm::LogPrint("LHCInfoESAnalyzer") << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("LHCInfoRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogPrint("LHCInfoESAnalyzer") << "Record \"LHCInfoRcd"
                                         << "\" does not exist " << std::endl;
    }
    edm::LogPrint("LHCInfoESAnalyzer") << "got eshandle" << std::endl;
    edm::ESHandle<LHCInfo> sum = context.getHandle(m_LHCInfoToken);
    edm::LogPrint("LHCInfoESAnalyzer") << "got context" << std::endl;
    const LHCInfo* summary = sum.product();
    edm::LogPrint("LHCInfoESAnalyzer") << "got LHCInfo* " << std::endl;
    edm::LogPrint("LHCInfoESAnalyzer") << "print  result" << std::endl;
    edm::LogPrint("LHCInfoESAnalyzer") << *summary;
  }
  DEFINE_FWK_MODULE(LHCInfoESAnalyzer);
}  // namespace edmtest
