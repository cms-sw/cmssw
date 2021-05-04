#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"

class DemoRunHarvester : public DQMEDHarvester {
private:
  std::string target_;
  int ctr_ = 0;

public:
  explicit DemoRunHarvester(const edm::ParameterSet&);
  ~DemoRunHarvester() override {}

  void beginRun(const edm::Run& run, const edm::EventSetup& c) override;
  void dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) override;
};

DemoRunHarvester::DemoRunHarvester(const edm::ParameterSet& iConfig)
    : DQMEDHarvester(iConfig), target_(iConfig.getParameter<std::string>("target")) {}

void DemoRunHarvester::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

void DemoRunHarvester::dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {
  ig.setCurrentFolder(target_);
  MonitorElement* me = ig.get(target_ + "/EXAMPLE");
  me->getTH1()->Fill(3);

  ib.setCurrentFolder(target_ + "_runsummary");
  MonitorElement* out = ib.book1D("EXAMPLE", "EXAMPLE", 100, 0., 100.);
  out->setBinContent(5, me->getBinContent(5));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DemoRunHarvester);
