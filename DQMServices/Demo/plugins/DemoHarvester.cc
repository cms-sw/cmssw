#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"

class DemoHarvester : public DQMEDHarvester {
private:
  std::string target_;
  int ctr_ = 0;

public:
  explicit DemoHarvester(const edm::ParameterSet &);
  ~DemoHarvester() override {}

  void beginRun(const edm::Run &run, const edm::EventSetup &c) override;
  void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &ib,
                             DQMStore::IGetter &ig,
                             edm::LuminosityBlock const &lumi,
                             edm::EventSetup const &) override;
};

DemoHarvester::DemoHarvester(const edm::ParameterSet &iConfig)
    : DQMEDHarvester(iConfig), target_(iConfig.getParameter<std::string>("target")) {}

void DemoHarvester::beginRun(const edm::Run &run, const edm::EventSetup &iSetup) {}

void DemoHarvester::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) {
  ig.setCurrentFolder(target_);
  MonitorElement *me = ig.get(target_ + "/EXAMPLE");
  me->getTH1()->Fill(3);

  ib.setCurrentFolder(target_ + "_runsummary");
  MonitorElement *out = ib.book1D("EXAMPLE", "EXAMPLE", 100, 0., 100.);
  out->setBinContent(5, me->getBinContent(5));
}

void DemoHarvester::dqmEndLuminosityBlock(DQMStore::IBooker &ib,
                                          DQMStore::IGetter &ig,
                                          edm::LuminosityBlock const &lumi,
                                          edm::EventSetup const &) {
  ig.setCurrentFolder(target_);
  MonitorElement *me = ig.get(target_ + "/EXAMPLE");
  me->getTH1()->Fill(4);

  ctr_++;

  ib.setCurrentFolder(target_ + "_lumisummary");
  MonitorElement *out = ib.book1D("EXAMPLE", "EXAMPLE", 100, 0., 100.);
  out->setBinContent(ctr_, lumi.luminosityBlock());
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DemoHarvester);
