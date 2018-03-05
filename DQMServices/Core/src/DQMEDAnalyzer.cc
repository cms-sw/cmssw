#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

DQMEDAnalyzer::DQMEDAnalyzer() {
  lumiToken_ = produces<DQMToken,edm::Transition::EndLuminosityBlock>("endLumi");
  runToken_ = produces<DQMToken,edm::Transition::EndRun>("endRun");
}

void DQMEDAnalyzer::beginRun(edm::Run const& run, edm::EventSetup const& setup) 
{
  dqmBeginRun(run, setup);
  edm::Service<DQMStore>()->bookTransaction(
    [this, &run, &setup](DQMStore::IBooker & booker)
    {
      booker.cd();
      this->bookHistograms(booker, run, setup);
    },
    run.run(),
    run.moduleCallingContext()->moduleDescription()->id());
}

void DQMEDAnalyzer::endRun(edm::Run const& run, edm::EventSetup const& setup) 
{ }

void DQMEDAnalyzer::endRunProduce(edm::Run& run, edm::EventSetup const& setup) 
{
  edm::Service<DQMStore>()->cloneRunHistograms(
      run.run(),
      run.moduleCallingContext()->moduleDescription()->id());

  run.put(runToken_, std::make_unique<DQMToken>());
}

void DQMEDAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) 
{ }

void DQMEDAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) 
{ }

void DQMEDAnalyzer::endLuminosityBlockProduce(edm::LuminosityBlock & lumi, edm::EventSetup const& setup) 
{
  edm::Service<DQMStore>()->cloneLumiHistograms(
      lumi.run(),
      lumi.luminosityBlock(),
      lumi.moduleCallingContext()->moduleDescription()->id());

  lumi.put(lumiToken_, std::make_unique<DQMToken>());
}


void DQMEDAnalyzer::dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}

void DQMEDAnalyzer::analyze(edm::Event const&, edm::EventSetup const&) {}
void DQMEDAnalyzer::accumulate(edm::Event const& ev, edm::EventSetup const& es) 
{
  analyze(ev, es);
}

