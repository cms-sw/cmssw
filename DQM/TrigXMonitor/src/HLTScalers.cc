// $Id: HLTScalers.cc,v 1.13 2008/08/24 16:34:57 wittich Exp $
// 
// $Log: HLTScalers.cc,v $
// Revision 1.13  2008/08/24 16:34:57  wittich
// - rate calculation cleanups
// - fix error logging with LogDebug
// - report the actual lumi segment number that we think it might be
//
// Revision 1.12  2008/08/22 20:56:55  wittich
// - add client for HLT Scalers
// - Move rate calculation to HLTScalersClient and slim down the
//   filter-farm part of HLTScalers
//
// Revision 1.11  2008/08/15 15:39:45  wteo
// split hltScalers into smaller histos, calculate rates
//
// Revision 1.10  2008/08/01 14:37:14  bjbloom
// Added ability to specify which paths are cross-correlated
//
// Revision 1.9  2008/07/04 15:57:18  wittich
// - move histograms to HLT directory (was in L1T)
// - add counter for number of lumi sections
// - attempt to hlt label histo axes locally; disabled (it's illegible)
//

#include <iostream>


// FW
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"

// HLT
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/HLTenums.h"
#include "FWCore/Framework/interface/TriggerNames.h"


#include "DQM/TrigXMonitor/interface/HLTScalers.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;



HLTScalers::HLTScalers(const edm::ParameterSet &ps):
  dbe_(0),
  scalersException_(0),
  hltCorrelations_(0),
  detailedScalers_(0), 
  nProc_(0),
  nLumiBlocks_(0),
  trigResultsSource_( ps.getParameter< edm::InputTag >("triggerResults")),
  resetMe_(true),
  monitorDaemon_(ps.getUntrackedParameter<bool>("MonitorDaemon", false)),
  nev_(0), 
  nLumi_(0),
  currentRun_(-1)
{



  LogDebug("Status") << "HLTScalers: constructor...." ;

  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  if (dbe_ ) {
    dbe_->setCurrentFolder("HLT/HLTScalers");
  }
  

}



void HLTScalers::beginJob(const edm::EventSetup& c)
{
  LogDebug("Status") << "HLTScalers::beginJob()..." << std::endl;

  if (dbe_) {
    dbe_->setCurrentFolder("HLT/HLTScalers");


    nProc_ = dbe_->bookInt("nProcessed");
    nLumiBlocks_ = dbe_->bookInt("nLumiBlocks");

    // other ME's are now found on the first event of the new run, 
    // when we know more about the HLT configuration.
  
  }
  return;
}

void HLTScalers::analyze(const edm::Event &e, const edm::EventSetup &c)
{
  nProc_->Fill(++nev_);

  edm::Handle<TriggerResults> hltResults;
  bool b = e.getByLabel(trigResultsSource_, hltResults);
  if ( !b ) {
    edm::LogInfo("Product") << "getByLabel for TriggerResults failed"
			   << " with label " << trigResultsSource_;
    return;
  }
  TriggerNames names(*hltResults);
  
  
  int npath = hltResults->size();

  // on the first event of a new run we book new ME's
  if (resetMe_ ) {
    LogInfo("Parameter") << "analyze(): new run. dump path for this evt " 
			 << e.id() << ", \n"
			 << *hltResults ;
    // need to get maxModules dynamically
    int maxModules = 200;
    //int npaths=hltResults->size();

    dbe_->setCurrentFolder("HLT/HLTScalers");


    detailedScalers_ = dbe_->book2D("detailedHltScalers", "HLT Scalers", 
				    npath, -0.5, npath-0.5,
				    maxModules, 0, maxModules-1);
    scalers_ = dbe_->book1D("hltScalers", "HLT scalers",
			    npath, -0.5, npath-0.5);
  

    scalersException_ = dbe_->book1D("hltExceptions", "HLT Exception scalers",
			    npath, -0.5, npath-0.5);


    hltCorrelations_ = dbe_->book2D("hltCorrelations", "HLT Scalers", 
		         	npath, -0.5, npath-0.5,
				npath, -0.5, npath-0.5);

    resetMe_ = false;
    // save path names in DQM-accessible format
    int q =0;
    for ( TriggerNames::Strings::const_iterator 
	    j = names.triggerNames().begin();
	  j !=names.triggerNames().end(); ++j ) {
      
      LogDebug("Parameter") << q << ": " << *j ;
      char pname[256];
      snprintf(pname, 256, "path%03d", q);
      ++q;
      MonitorElement *e = dbe_->bookString(pname, *j);
      hltPathNames_.push_back(e);  // I don't ever use these....
    }
  } // end resetme_ - pseudo-end-run record
  


  for ( int i = 0; i < npath; ++i ) {
    // state returns 0 on ready, 1 on accept, 2 on fail, 3 on exception.
    // these are defined in HLTEnums.h
    for ( unsigned int j = 0; j < hltResults->index(i); ++j ) {
      detailedScalers_->Fill(i,j);
    }
    if ( hltResults->state(i) == hlt::Pass) {
      scalers_->Fill(i);
      for ( int j = i + 1; j < npath; ++j ) {
	if ( hltResults->state(j) == hlt::Pass) {
	  hltCorrelations_->Fill(i,j); // fill 
	  hltCorrelations_->Fill(j,i);
	}
      }
    }
    else if ( hltResults->state(i) == hlt::Exception) {
      scalersException_->Fill(i);
    }
  }
  
}


void HLTScalers::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				    const edm::EventSetup& c)
{
  // put this in as a first-pass for figuring out the rate
  // each lumi block is 93 seconds in length
  nLumiBlocks_->Fill(lumiSeg.id().luminosityBlock());
 
  LogDebug("Status") << "End of luminosity block." ;

}


/// BeginRun
void HLTScalers::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("Status") << "HLTScalers::beginRun, run "
		     << run.id();
  if ( currentRun_ != int(run.id().run()) ) {
    resetMe_ = true;
    currentRun_ = run.id().run();
  }
}

/// EndRun
void HLTScalers::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("Status") << "HLTScalers::endRun , run "
		     << run.id();
  if ( currentRun_ != int(run.id().run()) ) {
    resetMe_ = true;
    currentRun_ = run.id().run();
  }
}
