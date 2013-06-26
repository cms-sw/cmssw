#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"

using namespace std;
using namespace edm;

HLTTauPostProcessor::HLTTauPostProcessor( const edm::ParameterSet& ps ) 
{
    sourceModule_   = ps.getUntrackedParameter<std::string>("SourceModule");
    dqmBaseFolder_  = ps.getUntrackedParameter<std::string>("DQMBaseFolder");
    hltProcessName_ = ps.getUntrackedParameter<std::string>("HLTProcessName","HLT");
    L1MatchDr_      = ps.getUntrackedParameter<double>("L1MatchDeltaR",0.5);
    HLTMatchDr_     = ps.getUntrackedParameter<double>("HLTMatchDeltaR",0.2);
    runAtEndJob_    = ps.getUntrackedParameter<bool>("runAtEndJob",false);
    runAtEndRun_    = ps.getUntrackedParameter<bool>("runAtEndRun",true);
    hltMenuChanged_ = true;
    automation_     = HLTTauDQMAutomation(hltProcessName_, L1MatchDr_, HLTMatchDr_); 
    ps_             = ps;
}

HLTTauPostProcessor::~HLTTauPostProcessor()
{
}

void HLTTauPostProcessor::beginJob()
{
}

void HLTTauPostProcessor::beginRun( const edm::Run& iRun, const edm::EventSetup& iSetup )
{
    //Evaluate configuration for every new trigger menu
    if ( HLTCP_.init(iRun, iSetup, hltProcessName_, hltMenuChanged_) ) {
        if ( hltMenuChanged_ ) {
            processPSet(ps_);
        }
    } else {
        edm::LogWarning("HLTTauPostProcessor") << "HLT config extraction failure with process name '" << hltProcessName_ << "'";
    }
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
    while (!summaryPlotters_.empty()) delete summaryPlotters_.back(), summaryPlotters_.pop_back();
    
    //Read the configuration
    for ( unsigned int i = 0; i < setup_.size(); ++i ) {
        summaryPlotters_.push_back(new HLTTauDQMSummaryPlotter(setup_[i],dqmBaseFolder_));
    }
    
    for ( unsigned int i = 0; i < summaryPlotters_.size(); ++i ) {
        if (summaryPlotters_[i]->isValid()) summaryPlotters_[i]->plot();
    }
}

void HLTTauPostProcessor::processPSet( const edm::ParameterSet& pset ) {
    //Get parameters
    setup_ = pset.getParameter<std::vector<edm::ParameterSet> >("Setup");
    
    //Automatic Configuration
    automation_.AutoCompleteConfig( setup_, HLTCP_ );
}
