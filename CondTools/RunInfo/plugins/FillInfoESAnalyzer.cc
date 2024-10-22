#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/RunInfo/interface/FillInfo.h"
#include "CondFormats/DataRecord/interface/FillInfoRcd.h"

namespace edmtest {
  class FillInfoESAnalyzer : public edm::one::EDAnalyzer<> {
  private:
    const edm::ESGetToken<FillInfo, FillInfoRcd> m_FillInfoToken;

  public:
    explicit FillInfoESAnalyzer(edm::ParameterSet const& p) : m_FillInfoToken(esConsumes()) {
      edm::LogPrint("FillInfoESAnalyzer") << "FillInfoESAnalyzer" << std::endl;
    }
    explicit FillInfoESAnalyzer(int i) {
      edm::LogPrint("FillInfoESAnalyzer") << "FillInfoESAnalyzer " << i << std::endl;
    }
    ~FillInfoESAnalyzer() override { edm::LogPrint("FillInfoESAnalyzer") << "~FillInfoESAnalyzer " << std::endl; }
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  };

  void FillInfoESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    edm::LogPrint("FillInfoESAnalyzer") << "###FillInfoESAnalyzer::analyze" << std::endl;
    edm::LogPrint("FillInfoESAnalyzer") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    edm::LogPrint("FillInfoESAnalyzer") << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("FillInfoRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogPrint("FillInfoESAnalyzer") << "Record \"FillInfoRcd"
                                          << "\" does not exist " << std::endl;
    }
    edm::LogPrint("FillInfoESAnalyzer") << "got eshandle" << std::endl;
    edm::ESHandle<FillInfo> sum = context.getHandle(m_FillInfoToken);
    edm::LogPrint("FillInfoESAnalyzer") << "got context" << std::endl;
    const FillInfo* summary = sum.product();
    edm::LogPrint("FillInfoESAnalyzer") << "got FillInfo* " << std::endl;
    edm::LogPrint("FillInfoESAnalyzer") << "print  result" << std::endl;
    edm::LogPrint("FillInfoESAnalyzer") << *summary;
  }
  DEFINE_FWK_MODULE(FillInfoESAnalyzer);
}  // namespace edmtest
