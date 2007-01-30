#include "DQM/HLTEvF/interface/PathTimerInserter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"

using namespace std;

PathTimerInserter::PathTimerInserter(const edm::ParameterSet& pset)
{
  produces<HLTPerformanceInfo>();
}

PathTimerInserter::~PathTimerInserter()
{
}  

// Functions that gets called by framework every event
void PathTimerInserter::produce(edm::Event& e, edm::EventSetup const&)
{
  
  // warning: the trigger results will be cleared as a result of inserting 
  // this object into the event
  
  edm::Service<edm::service::PathTimerService> pts;
  std::auto_ptr<HLTPerformanceInfo> prod=pts->getInfo();

  e.put(prod);
}

