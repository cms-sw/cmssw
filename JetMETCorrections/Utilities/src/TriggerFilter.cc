#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/CodedException.h"


#include "JetMETCorrections/Utilities/interface/TriggerFilter.h"
using namespace std;
using namespace reco;
using namespace edm;

TriggerFilter::TriggerFilter(ParameterSet const& cfg) 
{
  triggerName_               = cfg.getParameter<std::string>("triggerName");
  triggerProcessName_        = cfg.getParameter<std::string>("triggerProcessName"); 
  triggerResultsTag_         = cfg.getParameter<edm::InputTag>("triggerResultsTag");
  DEBUG_                     = cfg.getParameter<bool>("DEBUG");
}
////////////////////////////////////////////////////////////////////
void TriggerFilter::beginJob(const edm::EventSetup &iSetup) 
{
  EventsIN_ = 0;
  EventsOUT_ = 0; 
}

bool TriggerFilter::beginRun(edm::Run  &,edm::EventSetup const&iSetup) 
{
  //must be done at beginRun and not only at beginJob, because 
  //trigger names are allowed to change by run.
  hltConfig_.init(triggerProcessName_);
  // diagnostics:
  if (DEBUG_)
    {
      std::cout<<"There are "<<hltConfig_.size()<<" HLT triggers:"<<std::endl;
      for(unsigned int i=0;i<hltConfig_.size();i++)
        std::cout<<hltConfig_.triggerName(i)<<std::endl;
    } 
  //selectd index of chosen triggeR:
  triggerIndex_=hltConfig_.triggerIndex(triggerName_);
  if (triggerIndex_==hltConfig_.size()){
    string errorMessage="Requested TriggerName does not exist! -- "+triggerName_+"\n";
    throw  cms::Exception("Configuration",errorMessage);
  }
  //return type: no, we do not filter whole runs.
  return true;
}

////////////////////////////////////////////////////////////////////
bool TriggerFilter::filter(edm::Event &event, const edm::EventSetup &iSetup)
{
  EventsIN_++;
  event.getByLabel(triggerResultsTag_,triggerResultsHandle_);
  if (!triggerResultsHandle_.isValid()){
    string errorMessage="Requested TriggerResult is not present in file! -- "+triggerName_+"\n";
    throw  cms::Exception("Configuration",errorMessage);
  }

  //check if configuration matches trigger results:
  assert(triggerResultsHandle_->size()==hltConfig_.size());
  
  bool result=triggerResultsHandle_->accept(triggerIndex_);
  if (result) EventsOUT_++;
  return result;
  
}
//////////////////////////////////////////////////////////////////////////////////////////

void TriggerFilter::endJob() 
{
  cout<<"Total Events Processed: "<<EventsIN_<<", Events Passing the "<<triggerName_<<" trigger: "<<EventsOUT_<<endl;
}
