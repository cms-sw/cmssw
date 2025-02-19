// $Id: HLTScalers.cc,v 1.31 2011/04/01 09:47:00 rekovic Exp $
// 
// $Log: HLTScalers.cc,v $
// Revision 1.31  2011/04/01 09:47:00  rekovic
// Check if stream A exists before quering for its PD content
//
// Revision 1.30  2011/03/30 21:44:03  fwyzard
// make sure HLTConfigProvider is used only if succesfully initialized
//
// Revision 1.29  2011/03/30 21:35:40  fwyzard
// make sure all members are initialized
//
// Revision 1.28  2011/03/29 09:46:03  rekovic
// clean vector pairPDPaths in beginRun and tidy up
//
// Revision 1.27  2011/03/24 18:35:38  rekovic
// Change name for pd histo
//
// Revision 1.26  2011/03/24 18:25:45  rekovic
// Add single 1D plot of streamA content
//
// Revision 1.25  2010/07/20 02:58:27  wmtan
// Add missing #include files
//
// Revision 1.24  2010/03/17 20:54:51  wittich
// add scalers that I manually reset on beginLumi
//
// Revision 1.23  2010/02/25 17:34:01  wdd
// Central migration of TriggerNames class interface
//
// Revision 1.22  2010/02/24 17:43:47  wittich
// - keep trying to get path names if it doesn't work first time
// - move the Bx histograms out of raw to the toplevel directory.
//
// Revision 1.21  2010/02/11 23:54:28  wittich
// modify how the monitoring histo is filled
//
// Revision 1.20  2010/02/11 00:11:08  wmtan
// Adapt to moved framework header
//
// Revision 1.19  2010/02/02 13:53:05  wittich
// fix duplicate histogram name
//
// Revision 1.18  2010/02/02 11:42:53  wittich
// new diagnostic histograms
//
// Revision 1.17  2009/11/20 00:39:12  lorenzo
// fixes
//
// Revision 1.16  2008/09/03 13:59:06  wittich
// make HLT DQM path configurable via python parameter,
// which defaults to HLT/HLTScalers_EvF
//
// Revision 1.15  2008/09/03 02:13:47  wittich
// - bug fix in L1Scalers
// - configurable dqm directory in L1SCalers
// - other minor tweaks in HLTScalers
//

#include <iostream>


// FW
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// HLT
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/HLTenums.h"
#include "FWCore/Common/interface/TriggerNames.h"


#include "DQM/TrigXMonitor/interface/HLTScalers.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;



HLTScalers::HLTScalers(const edm::ParameterSet &ps):
  folderName_(ps.getUntrackedParameter<std::string>("dqmFolder", "HLT/HLTScalers_EvF")),
  processname_(ps.getParameter<std::string>("processname")),
  pairPDPaths_(),
  trigResultsSource_(ps.getParameter<edm::InputTag>("triggerResults")),
  dbe_(0),
  scalersN_(0),
  scalersException_(0),
  hltCorrelations_(0),
  detailedScalers_(0), 
  nProc_(0),
  nLumiBlock_(0),
  hltBx_(0),
  hltBxVsPath_(0),
  hltOverallScaler_(0),
  hltOverallScalerN_(0),
  diagnostic_(0),
  resetMe_(true),
  sentPaths_(false),
  monitorDaemon_(ps.getUntrackedParameter<bool>("MonitorDaemon", false)),
  nev_(0), 
  nLumi_(0),
  currentRun_(-1)
{
  LogDebug("HLTScalers") << "HLTScalers: constructor...." ;

  dbe_ = Service<DQMStore>().operator->();
  if (dbe_ ) {
    dbe_->setVerbose(0);
    dbe_->setCurrentFolder(folderName_);
  }
}


void HLTScalers::beginJob(void)
{
  LogDebug("HLTScalers") << "HLTScalers::beginJob()..." << std::endl;

  if (dbe_) {
    std::string rawdir(folderName_ + "/raw");
    dbe_->setCurrentFolder(rawdir);

    nProc_      = dbe_->bookInt("nProcessed");
    nLumiBlock_ = dbe_->bookInt("nLumiBlock");
    diagnostic_ = dbe_->book1D("hltMerge", "HLT merging diagnostic", 
                               1, 0.5, 1.5);

    // fill for ever accepted event 
    hltOverallScaler_  = dbe_->book1D("hltOverallScaler", "HLT Overall Scaler", 
                                      1, 0.5, 1.5);
    hltOverallScalerN_ = dbe_->book1D("hltOverallScalerN", 
                                      "Reset HLT Overall Scaler", 1, 0.5, 1.5);
    
    // other ME's are now found on the first event of the new run, 
    // when we know more about the HLT configuration.
  }
}

void HLTScalers::analyze(const edm::Event &e, const edm::EventSetup &c)
{
  nProc_->Fill(++nev_);
  diagnostic_->setBinContent(1,1); // this ME is never touched - 
                                   // it just tells you how the merging is doing.

  edm::Handle<TriggerResults> hltResults;
  bool b = e.getByLabel(trigResultsSource_, hltResults);
  if ( !b ) {
    edm::LogInfo("HLTScalers") << "getByLabel for TriggerResults failed"
                               << " with label " << trigResultsSource_;
    return;
  }
  
  
  int npath = hltResults->size();
  unsigned int nPD = pairPDPaths_.size();

  // on the first event of a new run we book new ME's
  if (resetMe_ ) {
    LogInfo("HLTScalers") << "analyze(): new run. dump path for this evt " 
                          << e.id() << ", \n"
                          << *hltResults ;

    if (not dbe_)
      return;

    // need to get maxModules dynamically
    int maxModules = 200;

    std::string rawdir(folderName_ + "/raw");
    dbe_->setCurrentFolder(rawdir);

    scalersPD_          = dbe_->book1D("pdScalers", "PD scalers (stream A)",
                                       nPD, -0.5, nPD-0.5);
    detailedScalers_    = dbe_->book2D("detailedHltScalers", "HLT Scalers", 
                                       npath, -0.5, npath-0.5,
                                       maxModules, 0, maxModules-1);
    scalers_            = dbe_->book1D("hltScalers", "HLT scalers",
                                       npath, -0.5, npath-0.5);
    scalersN_           = dbe_->book1D("hltScalersN", "Reset HLT scalers",
                                       npath, -0.5, npath-0.5);
    scalersException_   = dbe_->book1D("hltExceptions", "HLT Exception scalers",
                                       npath, -0.5, npath-0.5);
    hltCorrelations_    = dbe_->book2D("hltCorrelations", "HLT Scalers", 
                                       npath, -0.5, npath-0.5,
                                       npath, -0.5, npath-0.5);

    // these two belong in top-level
    dbe_->setCurrentFolder(folderName_);
    hltBxVsPath_        = dbe_->book2D("hltBxVsPath", "HLT Accept vs Bunch Number", 
                                       3600, -0.5, 3599.5,
                                       npath, -0.5, npath-0.5);
    hltBx_              = dbe_->book1D("hltBx", "Bx of HLT Accepted Events ", 
                                       3600, -0.5, 3599.5);

    resetMe_ = false;
  } // end resetMe_ - pseudo-end-run record

  const edm::TriggerNames & trigNames = e.triggerNames(*hltResults);
  // for some reason this doesn't appear to work on the first event sometimes
  if ( ! sentPaths_ ) {
    const edm::TriggerNames & names = e.triggerNames(*hltResults);

    // save path names in DQM-accessible format
    int q = 0;
    for ( TriggerNames::Strings::const_iterator 
          j = names.triggerNames().begin();
          j !=names.triggerNames().end(); ++j ) {
      
      LogDebug("HLTScalers") << q << ": " << *j ;
      ++q;
      scalers_->getTH1()->GetXaxis()->SetBinLabel(q, j->c_str());
    }

    for (unsigned int i = 0; i < nPD; i++) {
      LogDebug("HLTScalers") << i << ": " << pairPDPaths_[i].first << std::endl ;
      scalersPD_->getTH1()->GetXaxis()->SetBinLabel(i+1, pairPDPaths_[i].first.c_str());
    }

    sentPaths_ = true;
  }
      
  bool accept = false;
  int bx = e.bunchCrossing();
  for ( int i = 0; i < npath; ++i ) {
    // state returns 0 on ready, 1 on accept, 2 on fail, 3 on exception.
    // these are defined in HLTEnums.h
    for ( unsigned int j = 0; j < hltResults->index(i); ++j ) {
      detailedScalers_->Fill(i,j);
    }
    if ( hltResults->state(i) == hlt::Pass) {
      scalers_->Fill(i);
      scalersN_->Fill(i);
      hltBxVsPath_->Fill(bx, i);
      accept = true;
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
  if ( accept ) {
    hltOverallScaler_->Fill(1.0);
    hltOverallScalerN_->Fill(1.0);
    hltBx_->Fill(int(bx));
  }

  bool anyGroupPassed = false;
  for (unsigned int mi = 0; mi < pairPDPaths_.size(); mi++) {

    bool groupPassed = false;

    for (unsigned int i = 0; i < pairPDPaths_[mi].second.size(); i++)
    { 

      //string hltPathName =  hist_2d->GetXaxis()->GetBinLabel(i);
      std::string hltPathName =  pairPDPaths_[mi].second[i];

      // check if this is hlt path name
      //unsigned int pathByIndex = triggerNames.triggerIndex(hltPathName);
      unsigned int pathByIndex = trigNames.triggerIndex(pairPDPaths_[mi].second[i]);
      if(pathByIndex >= hltResults->size() ) continue;

      // check if its L1 passed
      // comment out below but set groupL1Passed to true always
      //if(hasL1Passed(hltPathName,triggerNames)) groupL1Passed = true;
      //groupL1Passed = true;

      // Fill HLTPassed Matrix and HLTPassFail Matrix
      // --------------------------------------------------------

      if(hltResults->accept(pathByIndex)) {
        groupPassed = true; 
        break;
     }

    }

    if(groupPassed) {
      scalersPD_->Fill(mi);
      anyGroupPassed = true;
    }

  }

  if(anyGroupPassed) scalersPD_->Fill(pairPDPaths_.size()-1);
}

void HLTScalers::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                                      const edm::EventSetup& c)
{
  LogDebug("HLTScalers") << "Start of luminosity block." ;
  // reset the N guys
  if ( scalersN_ ) 
    scalersN_->Reset();
  if ( hltOverallScalerN_ )
    hltOverallScalerN_->Reset();
}


void HLTScalers::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                                    const edm::EventSetup& c)
{
  // put this in as a first-pass for figuring out the rate
  // each lumi block is 23 seconds in length
  nLumiBlock_->Fill(lumiSeg.id().luminosityBlock());
 
  LogDebug("HLTScalers") << "End of luminosity block." ;

}


/// BeginRun
void HLTScalers::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("HLTScalers") << "HLTScalers::beginRun, run "
                         << run.id();
  if ( currentRun_ != int(run.id().run()) ) {
    resetMe_ = true;
    currentRun_ = run.id().run();
  }

  // HLT config does not change within runs!
  bool changed=false;

  // clear vector pairPDPaths_
  pairPDPaths_.clear();

  if (not hltConfig_.init(run, c, processname_, changed)) {
    edm::LogError("TrigXMonitor") << "HLTConfigProvider failed to initialize.";
  } else {

    // check if trigger name in (new) config
    //  cout << "Available TriggerNames are: " << endl;
    //  hltConfig_.dump("Triggers");

    if (hltConfig_.streamIndex("A")<hltConfig_.streamNames().size()) {

      // get hold of PD names and constituent path names
      const std::vector<std::string> & PD = hltConfig_.streamContent("A") ;

      for (unsigned int i = 0; i < PD.size(); i++) {

        const std::vector<std::string> & datasetPaths = hltConfig_.datasetContent(PD[i]);
        pairPDPaths_.push_back(make_pair(PD[i], datasetPaths));

      }

      // push stream A and its PDs
      pairPDPaths_.push_back(make_pair("A", PD));

    } else {

      LogDebug("HLTScalers") << "HLTScalers::beginRun, steamm A not in the HLT menu ";

    }

  }
}

/// EndRun
void HLTScalers::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("HLTScalers") << "HLTScalers::endRun , run "
                     << run.id();
  if ( currentRun_ != int(run.id().run()) ) {
    resetMe_ = true;
    currentRun_ = run.id().run();
  }
}
