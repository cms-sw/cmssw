#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"

#include "CondFormats/DataRecord/interface/L1TriggerScalerRcd.h"

namespace edmtest {
  class L1TriggerScalerESAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
  public:
    explicit L1TriggerScalerESAnalyzer(edm::ParameterSet const& p)
        : scaler1Token_(esConsumes<L1TriggerScaler, L1TriggerScalerRcd, edm::Transition::BeginRun>()),
          scaler2Token_(esConsumes<L1TriggerScaler, L1TriggerScalerRcd>()) {
      edm::LogVerbatim("L1TriggerScaler") << "L1TriggerScalerESAnalyzer";
    }
    explicit L1TriggerScalerESAnalyzer(int i)
        : scaler1Token_(esConsumes<L1TriggerScaler, L1TriggerScalerRcd, edm::Transition::BeginRun>()),
          scaler2Token_(esConsumes<L1TriggerScaler, L1TriggerScalerRcd>()) {
      edm::LogVerbatim("L1TriggerScaler") << "L1TriggerScalerESAnalyzer " << i;
    }
    ~L1TriggerScalerESAnalyzer() override { edm::LogVerbatim("L1TriggerScaler") << "~L1TriggerScalerESAnalyzer "; }
    void beginJob() override;
    void beginRun(const edm::Run&, const edm::EventSetup& context) override;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    void endRun(const edm::Run&, const edm::EventSetup& context) override {}

  private:
    const edm::ESGetToken<L1TriggerScaler, L1TriggerScalerRcd> scaler1Token_;
    const edm::ESGetToken<L1TriggerScaler, L1TriggerScalerRcd> scaler2Token_;
  };

  void L1TriggerScalerESAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& context) {
    edm::LogVerbatim("L1TriggerScaler") << "###L1TriggerScalerESAnalyzer::beginRun";
    const edm::ESHandle<L1TriggerScaler>& L1TriggerScaler_lumiarray = context.getHandle(scaler1Token_);
    edm::LogVerbatim("L1TriggerScaler") << " got eshandle with flag " << L1TriggerScaler_lumiarray.isValid()
                                        << " got data";
  }

  void L1TriggerScalerESAnalyzer::beginJob() {
    edm::LogVerbatim("L1TriggerScaler") << "###L1TriggerScalerESAnalyzer::beginJob";
  }

  void L1TriggerScalerESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    edm::LogVerbatim("L1TriggerScaler") << "###L1TriggerScalerESAnalyzer::analyze";

    // Context is not used.
    edm::LogVerbatim("L1TriggerScaler") << " I AM IN RUN NUMBER " << e.id().run();
    edm::LogVerbatim("L1TriggerScaler") << " ---EVENT NUMBER " << e.id().event();
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("L1TriggerScalerRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogVerbatim("L1TriggerScaler") << "Record \"L1TriggerScalerRcd"
                                          << "\" does not exist ";
    }
    const edm::ESHandle<L1TriggerScaler>& l1tr = context.getHandle(scaler2Token_);
    edm::LogVerbatim("L1TriggerScaler") << " got eshandle\n got context";
    const L1TriggerScaler* l1lumiscaler = l1tr.product();
    edm::LogVerbatim("L1TriggerScaler") << "got L1TriggerScaler* ";

    edm::LogVerbatim("L1TriggerScaler") << "print  result";
    l1lumiscaler->printRunValue();
    l1lumiscaler->printLumiSegmentValues();
    l1lumiscaler->printFormat();
    l1lumiscaler->printGTAlgoCounts();
    l1lumiscaler->printGTAlgoRates();
    l1lumiscaler->printGTAlgoPrescaling();
    l1lumiscaler->printGTTechCounts();
    l1lumiscaler->printGTTechRates();
    l1lumiscaler->printGTTechPrescaling();
    l1lumiscaler->printGTPartition0TriggerCounts();
    l1lumiscaler->printGTPartition0TriggerRates();
    l1lumiscaler->printGTPartition0DeadTime();
    l1lumiscaler->printGTPartition0DeadTimeRatio();
    edm::LogVerbatim("L1TriggerScaler") << "print  finished";
  }
  DEFINE_FWK_MODULE(L1TriggerScalerESAnalyzer);
}  // namespace edmtest
