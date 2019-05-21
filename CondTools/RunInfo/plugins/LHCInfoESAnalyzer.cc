#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

namespace edmtest {
  class LHCInfoESAnalyzer : public edm::EDAnalyzer {
  public:
    explicit LHCInfoESAnalyzer(edm::ParameterSet const& p) { std::cout << "LHCInfoESAnalyzer" << std::endl; }
    explicit LHCInfoESAnalyzer(int i) { std::cout << "LHCInfoESAnalyzer " << i << std::endl; }
    ~LHCInfoESAnalyzer() override { std::cout << "~LHCInfoESAnalyzer " << std::endl; }
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  };

  void LHCInfoESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    std::cout << "###LHCInfoESAnalyzer::analyze" << std::endl;
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("LHCInfoRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout << "Record \"LHCInfoRcd"
                << "\" does not exist " << std::endl;
    }
    edm::ESHandle<LHCInfo> sum;
    std::cout << "got eshandle" << std::endl;
    context.get<LHCInfoRcd>().get(sum);
    std::cout << "got context" << std::endl;
    const LHCInfo* summary = sum.product();
    std::cout << "got LHCInfo* " << std::endl;
    std::cout << "print  result" << std::endl;
    std::cout << *summary;
  }
  DEFINE_FWK_MODULE(LHCInfoESAnalyzer);
}  // namespace edmtest
