#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"

using namespace std;
using namespace edm;

HLTTauPostProcessor::HLTTauPostProcessor(const edm::ParameterSet& ps):
  runAtEndJob_(ps.getUntrackedParameter<bool>("runAtEndJob",false)),
  runAtEndRun_(ps.getUntrackedParameter<bool>("runAtEndRun",true))
{
  std::string dqmBaseFolder = ps.getUntrackedParameter<std::string>("DQMBaseFolder");
  for(const edm::ParameterSet& pset: ps.getParameter<std::vector<edm::ParameterSet> >("Setup")) {
    summaryPlotters_.emplace_back(new HLTTauDQMSummaryPlotter(pset, dqmBaseFolder));
  }
}

HLTTauPostProcessor::~HLTTauPostProcessor()
{
}

void HLTTauPostProcessor::analyze( const Event& iEvent, const EventSetup& iSetup )
{
}

void HLTTauPostProcessor::endRun( const Run& r, const EventSetup& context )
{
    if (runAtEndRun_) harvest();
}

void HLTTauPostProcessor::endJob()
{
    if (runAtEndJob_) harvest();
}

void HLTTauPostProcessor::harvest()
{    
  for(auto& plotter: summaryPlotters_) {
    if(plotter->isValid()) {
      plotter->bookPlots();
      plotter->plot();
    }
  }
}
