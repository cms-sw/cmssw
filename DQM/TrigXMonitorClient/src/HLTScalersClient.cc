// $Id: HLTScalersClient.cc,v 1.1 2008/08/22 20:56:57 wittich Exp $
// 
// $Log: HLTScalersClient.cc,v $
// Revision 1.1  2008/08/22 20:56:57  wittich
// - add client for HLT Scalers
// - Move rate calculation to HLTScalersClient and slim down the
//   filter-farm part of HLTScalers
//

#include <cassert>

#include "DQM/TrigXMonitorClient/interface/HLTScalersClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using edm::LogInfo;
using edm::LogWarning;

#define SECS_PER_LUMI_SECTION 93.3


/// Constructors
HLTScalersClient::HLTScalersClient(const edm::ParameterSet& ps):
  dbe_(0),
  nLumi_(0),
  currentRate_(0),
  currentLumiBlockNumber_(0)
{
  LogDebug("Status") << "constructor" ;
  // get back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  assert(dbe_ != 0); // blammo!
  dbe_->setCurrentFolder("HLT/HLTScalers");

  currentRate_ = dbe_->book1D("cur_rate", 
			      "current lumi section rate per path",
			      MAX_PATHS, -0.5, MAX_PATHS-0.5);

  for (int i = 0; i < MAX_PATHS; ++i ) {
    scalerCounters_[i] = 0UL;
    rateHistories_[i] = 0; // not really needed but ...
    char name[256]; snprintf(name, 256, "rate_p%03d", i);
    LogDebug("Parameter") << "name " << i << " is " << name ;
    rateHistories_[i] = dbe_->book1D(name, name, MAX_LUMI_SEG, 
				     -0.5, MAX_LUMI_SEG-0.5);
  }
  
}


/// BeginJob
void HLTScalersClient::beginJob(const edm::EventSetup& c)
{
  LogDebug("Status") << "beingJob" ;
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
  std::cout << "--------------> in endLumiBlock" << std::endl;
  nLumi_ = lumiSeg.id().luminosityBlock();
  MonitorElement *scalers = dbe_->get("HLT/HLTScalers/hltScalers");
  if ( scalers == 0 ) {
    LogInfo("Status") << "cannot get hlt scalers histogram, bailing out.";
    return;
  }

  int npaths = scalers->getNbinsX();
  if ( npaths > MAX_PATHS ) npaths = MAX_PATHS; // HARD CODE FOR NOW
  std::cout << "--------------> in endLumiBlock 2 " 
	    << nLumi_
	    << std::endl;

  MonitorElement *nLumi = dbe_->get("HLT/HLTScalers/nLumiBlocks");
  LogDebug("Parameter") << "Lumi Block from DQM: "
			<< (nLumi!=0?nLumi->getIntValue():-1)
			<< ", local is " << nLumi_;
  int nL = (nLumi!=0?nLumi->getIntValue():nLumi_);
  if ( nL > MAX_LUMI_SEG ) {
    LogWarning("Status") << "Too many Lumi segments, "
			 << nL << " is greater than MAX_LUMI_SEG";
    nL = MAX_LUMI_SEG;
  }
  float delta_t = (nL - currentLumiBlockNumber_)*SECS_PER_LUMI_SECTION;
  // fill in the rates
  for ( int i = 1; i <= npaths; ++i ) { // bins start at 1
    float current_count = scalers->getBinContent(i);
    float rate = (current_count-scalerCounters_[i-1])/delta_t;
    if ( rate > 1E-3 ) {
      LogDebug("Parameter") << "rate path " << i << " is " << rate;
    }
    currentRate_->setBinContent(i, rate);
    //currentRate_->setBinError(i, error);
    scalerCounters_[i-1] = ulong(current_count);
    
    rateHistories_[i-1]->setBinContent(nL, rate);
  }
  currentLumiBlockNumber_ = nL;

//   MonitorElement *l1scalers = dbe_->get("HLT/HLTScalers/l1Scalers");
//   // check which of the histograms are empty
}

// unused
void HLTScalersClient::analyze(const edm::Event& e, const edm::EventSetup& c) 
{
}
