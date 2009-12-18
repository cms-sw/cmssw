#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"
using namespace std;
using namespace edm;

//
// constructors and destructor
//
HLTTauPostProcessor::HLTTauPostProcessor( const edm::ParameterSet& ps ) 
{
  //Get General Monitoring Parameters
  config_= ps.getParameter<edm::ParameterSet>("Harvester");
  runAtEndJob_ = ps.getUntrackedParameter<bool>("runAtEndJob",false);
  runAtEndRun_ = ps.getUntrackedParameter<bool>("runAtEndRun",true);

}

HLTTauPostProcessor::~HLTTauPostProcessor()
{
}
void 
HLTTauPostProcessor::beginJob(){
}

void HLTTauPostProcessor::beginRun(const edm::Run& r, const edm::EventSetup& context) {
}

//--------------------------------------------------------
void HLTTauPostProcessor::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				      const EventSetup& context) {
}

// ----------------------------------------------------------
void 
HLTTauPostProcessor::analyze(const Event& iEvent, const EventSetup& iSetup )
{  }




//--------------------------------------------------------
void HLTTauPostProcessor::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
}
//--------------------------------------------------------
void HLTTauPostProcessor::endRun(const Run& r, const EventSetup& context){
  if(runAtEndRun_)
    harvest();

}
//--------------------------------------------------------
void HLTTauPostProcessor::endJob(){
  if(runAtEndJob_)
    harvest();
}


void HLTTauPostProcessor::harvest()
{
  HLTTauDQMSummaryPlotter summaryPlotter(config_);
  summaryPlotter.plot();
  return;
}
