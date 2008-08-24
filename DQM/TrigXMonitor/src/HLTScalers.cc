// $Id: HLTScalers.cc,v 1.12 2008/08/22 20:56:55 wittich Exp $
// 
// $Log: HLTScalers.cc,v $
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

// L1
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DQM/TrigXMonitor/interface/HLTScalers.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;



HLTScalers::HLTScalers(const edm::ParameterSet &ps):
  dbe_(0),
  scalersException_(0),
  hltCorrelations_(0),
  detailedScalers_(0), l1scalers_(0), 
  l1Correlations_(0),
  nProc_(0),
  nLumiBlocks_(0),
  trigResultsSource_( ps.getParameter< edm::InputTag >("triggerResults")),
  l1GtDataSource_( ps.getParameter< edm::InputTag >("l1GtData")),
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

    // fixed - only for 128 algo bits right now
    l1scalers_ = dbe_->book1D("l1Scalers", "L1 scalers (locally derived)",
			      128, -0.5, 127.5);
    l1Correlations_ = dbe_->book2D("l1Correlations", "L1 scaler correlations"
				   " (locally derived)", 
				   128, -0.5, 127.5,
				   128, -0.5, 127.5);
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

    l1scalers_->Reset(); // should never have any effect?
    l1Correlations_->Reset(); // should never have any effect?
    resetMe_ = false;
    // save path names in DQM-accessible format
    int q =0;
    for ( TriggerNames::Strings::const_iterator 
	    j = names.triggerNames().begin();
	  j !=names.triggerNames().end(); ++j ) {
      
      LogDebug("Parameter") << q << ": " << *j ;
      char pname[256];
      snprintf(pname, 256, "path%02d", q);
      // setting these here is a nice idea but it's totally illegible
      //scalers_[q/20]->setBinLabel(q+1, *j); 
      ++q;
      MonitorElement *e = dbe_->bookString(pname, *j);
      hltPathNames_.push_back(e);  // I don't ever use these....
    }
  } // end resetme_ - pseudo-end-run record
  
//   unsigned int n=0;
//   if(specifyPaths_)
//     {
//       n = pathNames_.size();
//       for (unsigned int i=0; i!=n; i++) {
//      	pathNamesIndex_.push_back(names.triggerIndex(pathNames_[i]));
//       }
//     }


  for ( int i = 0; i < npath; ++i ) {
    // state returns 0 on ready, 1 on accept, 2 on fail, 3 on exception.
    // these are defined in HLTEnums.h
//     LogDebug("Parameter") << "i = " << i << ", result = " 
// 			  << hltResults->accept(i)
// 			  << ", index = " << hltResults->index(i) ;
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
  
  // now the L1 equivalent. snarfed from L1TGT.cc
  // get Global Trigger decision and the decision word
  // these are locally derived; the HW scaler information is 
  // the definitive source for this data.
  edm::Handle<L1GlobalTriggerReadoutRecord> myGTReadoutRecord;
  bool t = e.getByLabel(l1GtDataSource_,myGTReadoutRecord);
  if ( ! t ) {
    edm::LogInfo("Product") << "can't find L1GlobalTriggerReadoutRecord "
			   << "with label " << l1GtDataSource_.label() ;
  }
  else {
    // vector of bool
    DecisionWord gtDecisionWord = myGTReadoutRecord->decisionWord();
    if ( ! gtDecisionWord.empty() ) { // if board not there this is zero
      for ( int i = 0; i < 127; ++i ) {
	if ( gtDecisionWord[i] ) {
	  l1scalers_->Fill(i);
	  for ( int j = i + 1; j < 127; ++j ) {
	    if ( gtDecisionWord[j] )
	      l1Correlations_->Fill(i,j);
	      l1Correlations_->Fill(j,i);
	  }
	}
      }
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
