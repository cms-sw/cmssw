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

}

HLTTauPostProcessor::~HLTTauPostProcessor()
{
}
void 
HLTTauPostProcessor::beginJob(const EventSetup& context){
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
}
//--------------------------------------------------------
void HLTTauPostProcessor::endJob(){
  HLTTauDQMSummaryPlotter summaryPlotter(config_);
  summaryPlotter.plot();
  return;
}


