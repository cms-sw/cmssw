// $Id: HLTScalers.cc,v 1.8 2008/03/01 00:40:16 lat Exp $
// 
// $Log: HLTScalers.cc,v $
// Revision 1.8  2008/03/01 00:40:16  lat
// DQM core migration.
//
// Revision 1.7  2008/01/22 19:02:43  muzaffar
// include cleanup. Only for cc/cpp files
//
// Revision 1.6  2007/12/11 17:24:55  wittich
// - add extra monitoring histos (eg hlt exceptions and correlations)
//
// Revision 1.5  2007/12/04 20:24:33  wittich
// - make hlt histograms variable width depending on path
// - add strings for path names
// - add int for nprocessed
// - add L1 scaler locally derived on Kaori's suggestion
//   + updates to cfi file for this, need to include unpacking of GT
//
// Revision 1.4  2007/12/01 19:28:56  wittich
// - fix cfi file (debug -> verbose, HLT -> FU for TriggerResults  label)
// - handle multiple beginRun for same run (don't call reset on DQM )
// - remove PathTimerService from cfg file in test subdir
//
// Revision 1.3  2007/11/26 19:43:37  wittich
// add cfi, add Reset() on endRun, cfg add tweaks
//
// Revision 1.2  2007/11/26 18:24:21  wittich
// fix bug in cfg handling
//
// Revision 1.1  2007/11/26 16:37:50  wittich
// Prototype HLT scaler information.
//

#include <iostream>


// FW
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  scalers_(0), scalersException_(0),
  hltCorrelations_(0),
  detailedScalers_(0), l1scalers_(0), 
  l1Correlations_(0),
  nProc_(0),
  nLumiBlocks_(0),
  trigResultsSource_( ps.getParameter< edm::InputTag >("triggerResults")),
  l1GtDataSource_( ps.getParameter< edm::InputTag >("l1GtData")),
  resetMe_(true),
  verbose_(ps.getUntrackedParameter < bool > ("verbose", false)),
  monitorDaemon_(ps.getUntrackedParameter<bool>("MonitorDaemon", false)),
  nev_(0), 
  nLumi_(0),
  currentRun_(-1)
{
  if ( verbose_ ) {
    std::cout << "HLTScalers::HLTScalers(ParameterSet) called...." 
	      << std::endl;
  }

  if (verbose_)
    std::cout << "HLTScalers: constructor...." << std::endl;

  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  if (dbe_ ) {
    dbe_->setCurrentFolder("HLT/HLTScalers");
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
    edm::LogInfo("HLTScalers") << "HLTScalers::analyze: getByLabel"
			       << " for TriggerResults failed"
			       << " with label " << trigResultsSource_;
    return;
  }
  
  
  int npath = hltResults->size();
  if ( verbose_ ) {
    std::cout << "HLTScalers::analyze: npath = " << npath << std::endl;
  }

  // on the first event of a new run we book new ME's
  if (resetMe_ ) {
    if ( verbose_) {
      std::cout << "HLTScalers::analyze(): new run. dump path for this evt " 
		<< e.id()
		<< std::endl;
      std::cout << *hltResults << std::endl;
      std::cout << std::endl;
    }
    // need to get this dynamically
    int maxModules = 200;
    int npaths=hltResults->size();

    detailedScalers_ = dbe_->book2D("detailedHltScalers", "HLT Scalers", 
				    npaths, -0.5, npaths-0.5,
				    maxModules, 0, maxModules-1);
    scalers_ = dbe_->book1D("hltScalers", "HLT scalers",
			    npaths, -0.5, npaths-0.5);
    scalersException_ = dbe_->book1D("hltExceptions", "HLT Exception scalers",
			    npaths, -0.5, npaths-0.5);

    hltCorrelations_ = dbe_->book2D("hltCorrelations", "HLT Scalers", 
				    npaths, -0.5, npaths-0.5,
				    npaths, -0.5, npaths-0.5);

    l1scalers_->Reset(); // should never have any effect?
    l1Correlations_->Reset(); // should never have any effect?
    resetMe_ = false;
    // save path names in DQM-accessible format
    TriggerNames names(*hltResults);
    int q =0;
    for ( TriggerNames::Strings::const_iterator 
	    j = names.triggerNames().begin();
	  j !=names.triggerNames().end(); ++j ) {
      if ( verbose_ )
	std::cout << q << ": " << *j << std::endl;
      char pname[256];
      snprintf(pname, 256, "path%02d", q);
      // setting these here is a nice idea but it's totally illegible
      //scalers_->setBinLabel(q+1, *j); 
      ++q;
      MonitorElement *e = dbe_->bookString(pname, *j);
      hltPathNames_.push_back(e);  // I don't ever use these....
    }
  } // end resetme_ - pseudo-end-run record

  for ( int i = 0; i < npath; ++i ) {
    if ( verbose_ ) {
      // state returns 0 on ready, 1 on accept, 2 on fail, 3 on exception.
      // these are defined in HLTEnums.h
      std::cout << "i = " << i << ", result = " << hltResults->state(i)
		<< ", index = " << hltResults->index(i) << std::endl;
    }
    for ( unsigned int j = 0; j < hltResults->index(i); ++j ) {
      detailedScalers_->Fill(i,j);
    }
    if ( hltResults->state(i) == hlt::Pass) {
      scalers_->Fill(i);
      // correlations
      for ( int j = i + 1; j < npath; ++j ) {
	if ( hltResults->state(j) == hlt::Pass ) {
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
    edm::LogInfo("HLTScalers") << "can't find L1GlobalTriggerReadoutRecord "
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
  nLumiBlocks_->Fill(++nLumi_);
}


/// BeginRun
void HLTScalers::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  if ( verbose_) {
    std::cout << "HLTScalers::beginRun, run "
	      << run.id()
	      << std::endl;
  }
  if ( currentRun_ != int(run.id().run()) ) {
    resetMe_ = true;
    currentRun_ = run.id().run();
  }
}

/// EndRun
void HLTScalers::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  if ( verbose_) {
    std::cout << "HLTScalers::endRun , run "
      	      << run.id()
	      << std::endl;
  }
  if ( currentRun_ != int(run.id().run()) ) {
    resetMe_ = true;
    currentRun_ = run.id().run();
  }
}


