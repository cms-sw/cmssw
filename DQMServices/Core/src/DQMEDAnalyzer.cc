#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

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
}


void DQMEDAnalyzer::dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}

void DQMEDAnalyzer::analyze(edm::Event const&, edm::EventSetup const&) {}
void DQMEDAnalyzer::accumulate(edm::Event const& ev, edm::EventSetup const& es) 
{
  analyze(ev, es);
}

