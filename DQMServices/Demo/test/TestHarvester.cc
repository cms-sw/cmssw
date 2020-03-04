#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class TestHarvester : public DQMEDHarvester {
private:
  std::string folder_;
  std::string whathappened;

public:
  explicit TestHarvester(const edm::ParameterSet &iConfig)
      : DQMEDHarvester(iConfig), folder_(iConfig.getParameter<std::string>("folder")) {}
  ~TestHarvester() override {}

  void beginRun(const edm::Run &run, const edm::EventSetup &iSetup) override {
    whathappened += "beginRun(" + std::to_string(run.run()) + ") ";
  }
  void dqmEndRun(DQMStore::IBooker &ib,
                 DQMStore::IGetter &ig,
                 const edm::Run &run,
                 const edm::EventSetup &iSetup) override {
    whathappened += "endRun(" + std::to_string(run.run()) + ") ";
    ig.setCurrentFolder(folder_);
    MonitorElement *out = ib.bookString("runsummary", "missing");
    out->Fill(whathappened);
  }

  void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) override {
    whathappened += "endJob() ";
    ig.setCurrentFolder(folder_);
    MonitorElement *out = ib.bookString("harvestingsummary", "missing");
    out->Fill(whathappened);
  }
  void dqmEndLuminosityBlock(DQMStore::IBooker &ib,
                             DQMStore::IGetter &ig,
                             edm::LuminosityBlock const &lumi,
                             edm::EventSetup const &) override {
    whathappened += "endLumi(" + std::to_string(lumi.run()) + "," + std::to_string(lumi.luminosityBlock()) + ") ";
  }
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "Harvesting/")->setComment("Where to put all the histograms");
    descriptions.add("testharvester", desc);
  }
};

DEFINE_FWK_MODULE(TestHarvester);

#include "FWCore/Framework/interface/EDAnalyzer.h"
class TestLegacyHarvester : public edm::EDAnalyzer {
private:
  std::string folder_;
  std::string whathappened;

public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit TestLegacyHarvester(const edm::ParameterSet &iConfig)
      : folder_(iConfig.getParameter<std::string>("folder")) {}
  ~TestLegacyHarvester() override {}

  void beginRun(const edm::Run &run, const edm::EventSetup &iSetup) override {
    whathappened += "beginRun(" + std::to_string(run.run()) + ") ";
  }
  void endRun(const edm::Run &run, const edm::EventSetup &iSetup) override {
    edm::Service<DQMStore> store;
    whathappened += "endRun(" + std::to_string(run.run()) + ") ";
    store->setCurrentFolder(folder_);
    MonitorElement *out = store->bookString("runsummary", "missing");
    out->Fill(whathappened);
  }

  void endJob() override {
    edm::Service<DQMStore> store;
    whathappened += "endJob() ";
    store->setCurrentFolder(folder_);
    MonitorElement *out = store->bookString("harvestingsummary", "missing");
    out->Fill(whathappened);
  }
  void endLuminosityBlock(edm::LuminosityBlock const &lumi, edm::EventSetup const &) override {
    whathappened += "endLumi(" + std::to_string(lumi.run()) + "," + std::to_string(lumi.luminosityBlock()) + ") ";
  }
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "LegacyHarvester/")->setComment("Where to put all the histograms");
    descriptions.add("testlegacyharvester", desc);
  }

  void analyze(edm::Event const &, edm::EventSetup const &) override {}
};

DEFINE_FWK_MODULE(TestLegacyHarvester);
