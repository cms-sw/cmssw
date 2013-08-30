#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"

using namespace std;
using namespace edm;

HLTTauPostProcessor::HLTTauPostProcessor( const edm::ParameterSet& ps ) 
{
    dqmBaseFolder_  = ps.getUntrackedParameter<std::string>("DQMBaseFolder");
    hltProcessName_ = ps.getUntrackedParameter<std::string>("HLTProcessName","HLT");
    L1MatchDr_      = ps.getUntrackedParameter<double>("L1MatchDeltaR",0.5);
    HLTMatchDr_     = ps.getUntrackedParameter<double>("HLTMatchDeltaR",0.2);
    runAtEndJob_    = ps.getUntrackedParameter<bool>("runAtEndJob",false);
    runAtEndRun_    = ps.getUntrackedParameter<bool>("runAtEndRun",true);
    hltMenuChanged_ = true;
    //Get parameters
    setup_ = ps.getParameter<std::vector<edm::ParameterSet> >("Setup");
}

HLTTauPostProcessor::~HLTTauPostProcessor()
{
}

void HLTTauPostProcessor::beginJob()
{
}

void HLTTauPostProcessor::beginRun( const edm::Run& iRun, const edm::EventSetup& iSetup )
{
}

void HLTTauPostProcessor::beginLuminosityBlock( const LuminosityBlock& lumiSeg, const EventSetup& context )
{
}

void HLTTauPostProcessor::analyze( const Event& iEvent, const EventSetup& iSetup )
{
}

void HLTTauPostProcessor::endLuminosityBlock( const LuminosityBlock& lumiSeg, const EventSetup& context )
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
    //Clear the plotter collection first
    summaryPlotters_.clear();
    
    //Read the configuration
    for ( unsigned int i = 0; i < setup_.size(); ++i ) {
        summaryPlotters_.emplace_back(new HLTTauDQMSummaryPlotter(setup_[i],dqmBaseFolder_));
    }
    
    for ( unsigned int i = 0; i < summaryPlotters_.size(); ++i ) {
        if (summaryPlotters_[i]->isValid()) summaryPlotters_[i]->plot();
    }
}
