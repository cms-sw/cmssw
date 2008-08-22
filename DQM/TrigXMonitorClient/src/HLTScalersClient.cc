// $Id: HLTScalers.cc,v 1.10 2008/08/01 14:37:14 bjbloom Exp $
// 
// $Log: HLTScalers.cc,v $

#include <cassert>

#include "DQM/TrigXMonitorClient/interface/HLTScalersClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using edm::LogInfo;

#define SECS_PER_LUMI_SECTION 93.0


/// Constructors
HLTScalersClient::HLTScalersClient(const edm::ParameterSet& ps):
  dbe_(0),
  currentRate_(0),
  currentLumiBlockNumber_(0)
{
  LogDebug("status") << "constructor" ;
  // get back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  assert(dbe_ != 0); // blammo!
  dbe_->setCurrentFolder("HLT/HLTScalers");

  for (int i = 0; i < MAX_PATHS; ++i ) {
    scalerCounters_[i] = 0UL;
    rateHistories_[i] = 0; // not really needed but ...
  }
  
}


/// BeginJob
void HLTScalersClient::beginJob(const edm::EventSetup& c)
{
  LogDebug("status") << "beingJob" ;
  if (dbe_) {
    dbe_->setCurrentFolder("HLT/HLTScalers");
  }
}


/// BeginRun
void HLTScalersClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
}

/// EndRun
void HLTScalersClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
}


/// End LumiBlock
/// DQM Client Diagnostic should be performed here
void HLTScalersClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			const edm::EventSetup& c)
{
  LogDebug("status") << "endLuminosityBlock";
  MonitorElement *scalers = dbe_->get("HLT/HLTScalers/hltScalers");
  if ( scalers == 0 ) {
    LogInfo("Status") << "cannot get hlt scalers histogram, bailing out.";
    return;
  }

  int npaths = scalers->getNbinsX();
  if ( npaths > MAX_PATHS ) npaths = MAX_PATHS; // HARD CODE FOR NOW

  MonitorElement *nLumi = dbe_->get("HLT/HLTScalers/nLumiBlocks");
  int nL = nLumi->getIntValue();
  float delta_t = (nL - currentLumiBlockNumber_)*SECS_PER_LUMI_SECTION;
  // fill in the rates
  for ( int i = 1; i <= npaths; ++i ) { // bins start at 1
    float current_count = scalers->getBinContent(i);
    float rate = (current_count-scalerCounters_[i-1])/delta_t;
    currentRate_->setBinContent(i, rate);
    //currentRate_->setBinError(i, error);
    scalerCounters_[i-1] = current_count;
    rateHistories_[i-1]->setBinContent(nL, rate);
  }

//   MonitorElement *l1scalers = dbe_->get("HLT/HLTScalers/l1Scalers");
//   // check which of the histograms are empty
}

// unused
void HLTScalersClient::analyze(const edm::Event& e, const edm::EventSetup& c) 
{
  LogDebug("status") << "analyze";

}
