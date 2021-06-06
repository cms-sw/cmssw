#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/DQMObjects/interface/DQMSummary.h"
#include "CondFormats/DataRecord/interface/DQMSummaryRcd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmtest {
  class DQMSummaryEventSetupAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit DQMSummaryEventSetupAnalyzer(const edm::ParameterSet& pset);
    explicit DQMSummaryEventSetupAnalyzer(int i);
    ~DQMSummaryEventSetupAnalyzer() override;
    void analyze(const edm::Event& event, const edm::EventSetup& setup) override;

  private:
    const edm::ESGetToken<DQMSummary, DQMSummaryRcd> dqmSummaryToken_;
  };

  DQMSummaryEventSetupAnalyzer::DQMSummaryEventSetupAnalyzer(const edm::ParameterSet& pset)
      : dqmSummaryToken_(esConsumes()) {
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "DQMSummaryEventSetupAnalyzer" << std::endl;
  }

  DQMSummaryEventSetupAnalyzer::DQMSummaryEventSetupAnalyzer(int i) : dqmSummaryToken_(esConsumes()) {
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "DQMSummaryEventSetupAnalyzer" << i << std::endl;
  }

  DQMSummaryEventSetupAnalyzer::~DQMSummaryEventSetupAnalyzer() {
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "~DQMSummaryEventSetupAnalyzer" << std::endl;
  }

  void DQMSummaryEventSetupAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "### DQMSummaryEventSetupAnalyzer::analyze" << std::endl;
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "--- RUN NUMBER: " << event.id().run() << std::endl;
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "--- EVENT NUMBER: " << event.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DQMSummaryRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      throw cms::Exception("Record not found") << "Record \"DQMSummaryRcd"
                                               << "\" does not exist!" << std::endl;
    }

    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "got EShandle" << std::endl;
    edm::ESHandle<DQMSummary> sum = setup.getHandle(dqmSummaryToken_);
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "got the Event Setup" << std::endl;
    const DQMSummary* summary = sum.product();
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "got DQMSummary* " << std::endl;
    edm::LogPrint("DQMSummaryEventSetupAnalyzer") << "print result" << std::endl;
    summary->printAllValues();
  }

  DEFINE_FWK_MODULE(DQMSummaryEventSetupAnalyzer);
}  // namespace edmtest
