// $Id: HLTScalers.cc,v 1.2 2007/11/26 18:24:21 wittich Exp $
// 
// $Log: HLTScalers.cc,v $
// Revision 1.2  2007/11/26 18:24:21  wittich
// fix bug in cfg handling
//
// Revision 1.1  2007/11/26 16:37:50  wittich
// Prototype HLT scaler information.
//

#include <iostream>


// FW
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/HLTenums.h"

#include "DQM/TrigXMonitor/interface/HLTScalers.h"
#include "DataFormats/Common/interface/Handle.h"

using namespace edm;



HLTScalers::HLTScalers(const edm::ParameterSet &ps):
  dbe_(0),
  scalers_(0), detailedScalers_(0),
  trigResultsSource_( ps.getParameter< edm::InputTag >("triggerResults")),
  resetMe_(true),
  verbose_(ps.getUntrackedParameter < bool > ("verbose", false)),
  monitorDaemon_(ps.getUntrackedParameter<bool>("MonitorDaemon", false))
{
  if ( verbose_ ) {
    std::cout << "HLTScalers::HLTScalers(ParameterSet) called...." 
	      << std::endl;
  }

  if (verbose_)
    std::cout << "HLTScalers: constructor...." << std::endl;

  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  dbe_->setVerbose(0);

  if ( monitorDaemon_) {
    Service<MonitorDaemon> daemon;
    daemon.operator->(); // gross
    monitorDaemon_ = true;
  }
  if (dbe_ ) {
    dbe_->setCurrentFolder("L1T/HLTScalers");
  }

}



void HLTScalers::beginJob(const edm::EventSetup& c)
{
  if ( verbose_ ) {
    std::cout << "HLTScalers::beginJob()..." << std::endl;
  }

  if (dbe_) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    }
    dbe_->setCurrentFolder("L1T/HLTScalers");

    // need to get these dynamically
    int maxModules = 200;
    int npaths=100;


    detailedScalers_ = dbe_->book2D("detailedHltScalers", "HLT Scalers", 
				    npaths, 0, npaths-1,
				    maxModules, 0, maxModules-1);
    scalers_ = dbe_->book1D("hltScalers", "HLT scalers",
			    npaths, -0.5, npaths-1.5);
  
  }
  return;
}

void HLTScalers::analyze(const edm::Event &e, const edm::EventSetup &c)
{
  edm::Handle<TriggerResults> results;
  bool b = e.getByLabel(trigResultsSource_, results);
  if ( !b ) {
    if ( verbose_ ) {
      std::cout << "HLTScalers::analyze: getByLabel failed with label " 
		<< trigResultsSource_
		<< std::endl;;
    }
    return;
  }
  
  
  int npath = results->size();
  if ( verbose_ ) {
    std::cout << "HLTScalers::analyze: npath = " << npath << std::endl;
  }

  // this is a scaler at the beginning of a run. I should
  // also dynamically set up the size of the histogram here.
  if (resetMe_ ) {
    if ( verbose_) {
      std::cout << "HLTScalers: new run: " << std::endl;
      std::cout << *results << std::endl;
      std::cout << std::endl;
      for ( int i = 0; i < npath; ++i ) {
	;
      }
    }
    scalers_->Reset();
    detailedScalers_->Reset();
    resetMe_ = false;
  }
  int nm = 0;
  for ( int i = 0; i < npath; ++i ) {
    if ( verbose_ ) {
      std::cout << "i = " << i << ", result = " << results->state(i)
		<< ", index = " << results->index(i) << std::endl;
    }
    nm += results->index(i);
    for ( unsigned int j = 0; j < results->index(i); ++j ) {
      detailedScalers_->Fill(i,j);
    }
    if ( results->state(i) == hlt::Pass) {
      scalers_->Fill(i);
    }
  }
  if ( verbose_ ) {
    std::cout << "np, nm = " 
	      << npath
	      << ", " 
	      << nm << std::endl;
  }
}


void HLTScalers::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				    const edm::EventSetup& c)
{
  // does nothing yet (TM)
}


/// BeginRun
void HLTScalers::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  if ( verbose_) {
    std::cout << "HLTScalers::beginRun "<< std::endl;
  }
  resetMe_ = true;
}

/// EndRun
void HLTScalers::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  if ( verbose_) {
    std::cout << "HLTScalers::endRun "<< std::endl;
  }
  resetMe_ = true;
}


