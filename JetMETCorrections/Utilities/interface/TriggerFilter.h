#ifndef __TriggerFilter_H__
#define __TriggerFilter_H__
#include "FWCore/Framework/interface/EDFilter.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"


class TriggerFilter : public edm::EDFilter 
{
  public:
    explicit TriggerFilter(const edm::ParameterSet & cfg);
    virtual void beginJob(const edm::EventSetup &iSetup);
    virtual bool beginRun(edm::Run &, edm::EventSetup const &iSetup);
    bool filter(edm::Event &event, const edm::EventSetup &iSetup);
    virtual void endJob();  
  private:
    std::string triggerName_;
    std::string triggerProcessName_;
    unsigned triggerIndex_;
    edm::Handle<edm::TriggerResults> triggerResultsHandle_;
    edm::InputTag triggerResultsTag_;
    HLTConfigProvider hltConfig_;
    bool DEBUG_;
    int EventsIN_,EventsOUT_;
};
#endif
