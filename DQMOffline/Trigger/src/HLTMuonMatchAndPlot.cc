 /** \file DQMOffline/Trigger/HLTMuonMatchAndPlot.cc
 *
 *  Muon HLT Offline DQM plotting code
 *  This object will make occupancy/efficiency plots for a
 *  specific set of conditions:
 *    1. A set of selection cuts
 *    2. A trigger name
 *  
 *  $Author: slaunwhj $
 *  $Date: 2010/03/16 14:35:40 $
 *  $Revision: 1.16 $
 */


#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlot.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// For storing calorimeter isolation info in the ntuple
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TPRegexp.h"
#include <iostream>

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;

typedef std::vector< edm::ParameterSet > Parameters;
typedef std::vector<reco::Muon> MuonCollection;

const int numCones     = 3;
const int numMinPtCuts = 1;
double coneSizes[] = { 0.20, 0.24, 0.30 };
double minPtCuts[] = { 0. };

const int HLTMuonMatchAndPlot::POS_CHARGE;
const int HLTMuonMatchAndPlot::NEG_CHARGE;

/// Constructor
HLTMuonMatchAndPlot::HLTMuonMatchAndPlot
( const ParameterSet& pset, string triggerName, vector<string> moduleNames,
  MuonSelectionStruct inputSelection, string customName,
  vector<string> validTriggers, edm::Run const& currentRun, edm::EventSetup const& currentEventSetup )
  :  mySelection(inputSelection), selectedValidTriggers(validTriggers)
{


  LogTrace ("HLTMuonVal") << "\n\n Inside HLTMuonMatchAndPlot Constructor";
  LogTrace ("HLTMuonVal") << "The trigger name is " << triggerName
                          << " and the module names are listed";

  for (vector<string>::iterator iMod = moduleNames.begin();
       iMod != moduleNames.end(); iMod++){
    LogTrace ("HLTMuonVal") << (*iMod);
  }
    
  
  theHltProcessName  = pset.getParameter<string>("HltProcessName");

  LogTrace ("HLTMuonVal") << "HLTMuonMatchAndPlot: Constructor: Initializing HLTConfigProvider with HLT process name: " << theHltProcessName << endl;

  
  HLTConfigProvider hltConfig;
  bool hltConfigChanged;
  bool hltConfigInitSuccess = hltConfig.init(currentRun, currentEventSetup, theHltProcessName, hltConfigChanged);

  theNumberOfObjects = ( TString(triggerName).Contains("Double") ) ? 2 : 1;
  theTriggerName     = triggerName;
    
  useAod         = true;

  requireL1SeedForHLTPaths = pset.getUntrackedParameter<bool>("RequireRecoToMatchL1Seed", false);

  matchType = pset.getUntrackedParameter<string>("matchType");

  // JMS Added a method to make standalone histogram output
  createStandAloneHistos = pset.getUntrackedParameter<bool>("createStandAloneHistos");
  histoFileName = pset.getUntrackedParameter<string> ("histoFileName");

  theHltCollectionLabels.clear();

  // make these config file inputs eventually
  useOldLabels = false;
  useFullDebugInformation = false;

  if (useFullDebugInformation) {
    // if you're filling all plots
    // then you'll have (All, L1, HLT)
    HLT_PLOT_OFFSET = 2;
  } else {
    // if you're only filling
    // the plots for one trigger
    // then have (All, HLT)
    HLT_PLOT_OFFSET = 1;
  }

  isL1Path = false;
  isL2Path = false;
  isL3Path = true;

  if (TString(triggerName).Contains("L1")) {
    isL1Path = true;
    isL3Path = false;
  }

  if (TString(triggerName).Contains("L2")) {
    isL2Path = true;
    isL3Path = false;
  }
  if (TString(triggerName).Contains("L3")) {
    isL3Path = true;
    isL3Path = false;
  }

    
  
  LogTrace ("HLTMuonVal") << "Parsing Module names... useOldLabels? "
                          << useOldLabels
                          << ",  useFullDebugInformation? "
                          << useFullDebugInformation
                          << endl;
                          
  
  TPRegexp l1Regexp("L1.*Filtered");
  TPRegexp l2Regexp("L2.*Filtered");

  string theLastHltFilter = "";

  theL1SeedModuleForHLTPath = "" ;


  for ( size_t i = 0; i < moduleNames.size(); i++ ) {
    string module = moduleNames[i];

    if (hltConfigInitSuccess) {
      LogTrace ("HLTMuonVal") << "Considering Module named    "
                              << module
                              << "      which has type =    "
                              << hltConfig.moduleType(module);

      if ( hltConfig.moduleType(module) == "HLTLevel1GTSeed" ) {
        LogTrace ("HLTMuonVal") << "Module = " << module
                                <<  " is a HLTLevel1GTSeed!!"
                                << endl
                                << "Storing it as L1Seed"
                                << endl;
        theL1SeedModuleForHLTPath = module;
      }
    }

    
    if ( TString(module).Contains(l1Regexp) ) {
      // this will be used to look up info
      // in the AOD information
      theL1CollectionLabel = module;
      LogTrace ("HLTMuonVal") << "... module is L1 collection";

      
    } else if ( TString(module).Contains(l2Regexp) ) {
      // this is a l2 module, only consider it if the trigger
      // is an L2 passthrough or if we want to use debug trigger info

      if (useFullDebugInformation) {
        theHltCollectionLabels.push_back(module);
        LogTrace ("HLTMuonVal") << "... module added to HLT collection list";
      } else if (isL2Path) {
        LogTrace ("HLTMuonVal") << "... module saved for later";
        theLastHltFilter = module;
      }

      
    } else if (TString(module).Contains("Filtered") ) {
      // must be L3 filtered here
      // potential to have Pre vs Iso filtered
      // In AOD, I think there is only information
      // for the *final* filter

      if (useFullDebugInformation) {
        LogTrace ("HLTMuonVal") << "... module added to HLT collection list" << endl;
        theHltCollectionLabels.push_back(module);        
      } else if (isL3Path) {
        LogTrace ("HLTMuonVal") << "... module saved for later" << endl;
        theLastHltFilter = module;
      }      

    }
  }

  if (!useFullDebugInformation && theLastHltFilter != "" ) {
    
    LogTrace("HLTMuonVal") << "\nAfter running over labels, we find hlt label coll size = "
                           << theHltCollectionLabels.size() << endl
                           << "\n\nwill only use the final hlt label = "
                           << theLastHltFilter << endl;

    theHltCollectionLabels.push_back (theLastHltFilter);
  }
  

  numHltLabels   = theHltCollectionLabels.size();
  isIsolatedPath = ( numHltLabels == 4 ) ? true : false;

  // -- Right now the new way is to hard-code it
  // -- this uses the most generic kind of muon
  // -- selectors will handle other cuts
  //theRecoLabel = "muons";

  RecoMuonInputTag =  pset.getParameter<edm::InputTag>("RecoMuonInputTag");  
  BeamSpotInputTag = pset.getParameter<edm::InputTag>("BeamSpotInputTag");
  HltRawInputTag  = pset.getParameter<edm::InputTag>("HltRawInputTag");
  HltAodInputTag = pset.getParameter<edm::InputTag>("HltAodInputTag");
  
  TriggerResultLabel = pset.getParameter<edm::InputTag>("TriggerResultLabel");  
  
  //useMuonFromGenerator = false; // = ( theGenLabel  == "" ) ? false : true;
  useMuonFromReco      = true; // = ( theRecoLabel == "" ) ? false : true;

  //define default parameters so that you don't crash

  vector <double> etaDefault;
  etaDefault.push_back(15);
  etaDefault.push_back(-2.1);
  etaDefault.push_back(2.1);

  theEtaParameters   = pset.getUntrackedParameter< vector<double> >("EtaParameters", etaDefault);
  
  vector <double> phiDefault;
  phiDefault.push_back(15);
  phiDefault.push_back(-3.2);
  phiDefault.push_back(3.2);

  thePhiParameters   = pset.getUntrackedParameter< vector<double> >("PhiParameters", phiDefault);

  // leave this vector of size 0
  vector <double> ptDefault;

  //  pt parameters are a different story
  //  it's a vector of doubles but it unpacked
  //  as bin low edges
  thePtParameters    = pset.getUntrackedParameter< vector<double> >("PtParameters", ptDefault);

  
  vector <double> resDefault;
  resDefault.push_back(10);
  resDefault.push_back(-0.1);
  resDefault.push_back(0.1);
  theResParameters = pset.getUntrackedParameter < vector<double> >("ResParameters", resDefault);  

  vector <double> d0Default;
  d0Default.push_back(10);
  d0Default.push_back(-2.0);
  d0Default.push_back(2.0);

  theD0Parameters = pset.getUntrackedParameter <vector<double> > ("D0Parameters", d0Default);

  vector <double> z0Default;
  z0Default.push_back(10);
  z0Default.push_back(-25);
  z0Default.push_back(25);

  theZ0Parameters = pset.getUntrackedParameter < vector<double> > ("Z0Parameters", z0Default);

  
  
  int numPtBinEdge = 0;
  if ( thePtParameters.size() > 100) {
    LogInfo ("HLTMuonVal") << "Warning!!! You specified a list of pt bin edges that is > 100 bins"
                           << "This is too many bins!! Truncating the list!!! " << endl;
    numPtBinEdge = 100;
  } else if (thePtParameters.size() < 1) {

    numPtBinEdge = 5;
    numBinsInPtHisto = numPtBinEdge - 1;
    ptBins[0] = 0;
    ptBins[1] = 20;
    ptBins[2] = 50;
    ptBins[3] = 100;
    ptBins[4] = 150;
    
  } else {
    numPtBinEdge = thePtParameters.size();
    // the number of bins in the histo is one
    // less than the number of edges
    numBinsInPtHisto = numPtBinEdge - 1;
  
    for (int iBin = 0; iBin < numPtBinEdge; iBin++){
      ptBins[iBin] = (float) thePtParameters[iBin];
      //LogTrace ("HLTMuonVal") << the 
    }

  }


  
  // Duplicate the pt parameters for some 2D histos
//   for(int i =0; i < 2; i++){
//     for (std::vector<double>::const_iterator iNum = theMaxPtParameters.begin();
//          iNum != theMaxPtParameters.end();
//          iNum++){
      
//       // if this is the # of bins, then
//       // halve the number of bins.
//       if (iNum == theMaxPtParameters.begin()){
//         theMaxPtParameters2d.push_back(floor((*iNum)/2));
//       } else {
//         theMaxPtParameters2d.push_back((*iNum));
//       }
//     }
//   }

//   // Duplicate the eta parameters for some 2D histos
//   for(int i =0; i < 2; i++){
//     for (std::vector<double>::const_iterator iNum = theEtaParameters.begin();
//          iNum != theEtaParameters.end();
//          iNum++){
//       // if this is the nBins param, halve it
//       if (iNum ==  theEtaParameters.begin()){
//         theEtaParameters2d.push_back(floor((*iNum)/2));      
//       } else {
//         theEtaParameters2d.push_back(*iNum);                   
//       }
      
//       // also fill the eta/phi plot parameters
//       // but don't worry about doubleing bins
//       // if (i < 1){
//       //         if (iNum ==  theEtaParameters.begin()){
//       //           thePhiEtaParameters2d.push_back(floor((*iNum)/2));      
//       //         } else {
//       //           thePhiEtaParameters2d.push_back(*iNum);      
      
//       //         } 
      
//       //       }
//     }
//   }

//   // Duplicate the pt parameters for some 2D histos
//   for(int i =0; i < 2; i++){
//     for (std::vector<double>::const_iterator iNum = thePhiParameters.begin();
//          iNum != thePhiParameters.end();
//          iNum++){

//       if (iNum == thePhiParameters.begin()) {
//         thePhiParameters2d.push_back(floor((*iNum)/2));
//       } else {
//         thePhiParameters2d.push_back(*iNum);
//       }

//       //       if (i < 1){
      
//       //         // if (iNum ==  theEtaParameters.begin()){
//       //         //           thePhiEtaParameters2d.push_back(floor((*iNum)/2));      
//       //         //         } else {
//       //         //           thePhiEtaParameters2d.push_back(*iNum);      
      
//       //         //         } 
      
//       //       }
//     }
//   }



  //=======================================



  theL1DrCut     = pset.getUntrackedParameter<double>("L1DrCut", 0.4);
  theL2DrCut     = pset.getUntrackedParameter<double>("L2DrCut", 0.1);
  theL3DrCut     = pset.getUntrackedParameter<double>("L3DrCut", 0.05);

  
  //==========================================
  // Hard-coded parameters
  // Make modifibly from script later
  //==========================================


  // put in the phi parameters
  thePhiEtaParameters2d.push_back(10);
  thePhiEtaParameters2d.push_back(-2.2);
  thePhiEtaParameters2d.push_back(2.2);

  thePhiEtaParameters2d.push_back(10);
  thePhiEtaParameters2d.push_back(-3.15);
  thePhiEtaParameters2d.push_back(3.15);
  
  
  // theD0Parameters.push_back(25);
  // theD0Parameters.push_back(-50.0);
  // theD0Parameters.push_back(50.0);
  
  //   theZ0Parameters.push_back(25);
  //   theZ0Parameters.push_back(-100);
  //   theZ0Parameters.push_back(100);

  theChargeParameters.push_back(3);
  theChargeParameters.push_back(-1.5);
  theChargeParameters.push_back(1.5);

  theDRParameters.push_back(10);
  theDRParameters.push_back(0.0);
  theDRParameters.push_back(theL2DrCut);

  theChargeFlipParameters.push_back(2);
  theChargeFlipParameters.push_back(-1.0);
  theChargeFlipParameters.push_back(1.0);
  theChargeFlipParameters.push_back(2);
  theChargeFlipParameters.push_back(-1.0);
  theChargeFlipParameters.push_back(1.0);

  theIsolationParameters.push_back(10);
  theIsolationParameters.push_back(0.0);
  theIsolationParameters.push_back(1.0);

  thePhiParameters0Pi.push_back(10);
  thePhiParameters0Pi.push_back(0);
  thePhiParameters0Pi.push_back(3.2);

  // theDeltaPhiVsPhiParameters.push_back(50);
  //   theDeltaPhiVsPhiParameters.push_back(-3.15);
  //   theDeltaPhiVsPhiParameters.push_back(3.15);
  //   theDeltaPhiVsPhiParameters.push_back(50);
  //   theDeltaPhiVsPhiParameters.push_back(0);
  //   theDeltaPhiVsPhiParameters.push_back(3.2);

//   theDeltaPhiVsZ0Parameters.push_back(theZ0Parameters[0]);
//   theDeltaPhiVsZ0Parameters.push_back(theZ0Parameters[1]);
//   theDeltaPhiVsZ0Parameters.push_back(theZ0Parameters[2]);
//   theDeltaPhiVsZ0Parameters.push_back(50);
//   theDeltaPhiVsZ0Parameters.push_back(0);
//   theDeltaPhiVsZ0Parameters.push_back(3.2);

//   theDeltaPhiVsD0Parameters.push_back(theD0Parameters[0]);
//   theDeltaPhiVsD0Parameters.push_back(theD0Parameters[1]);
//   theDeltaPhiVsD0Parameters.push_back(theD0Parameters[2]);
//   theDeltaPhiVsD0Parameters.push_back(50);
//   theDeltaPhiVsD0Parameters.push_back(0);
//   theDeltaPhiVsD0Parameters.push_back(3.2);
      
  

  
  dbe_ = 0 ;
  if ( pset.getUntrackedParameter<bool>("DQMStore", false) ) {
    dbe_ = Service<DQMStore>().operator->();
    dbe_->setVerbose(0);
  }

  if (!dbe_) {

    LogInfo ("HLTMuonVal") << "===WARNING=== Couldn't find DQMStore..." 
                           << "Won't be able to book ME's..."
                           << "The rest of the run will probably not be useful..."
                           << endl;

  }

  eventNumber = 0;

  LogTrace ("HLTMuonVal") << "exiting constructor\n\n";

}



void HLTMuonMatchAndPlot::endRun (const edm::Run& r, const edm::EventSetup& c)
{

  LogTrace ("HLTMuonVal") << "\n\nInside HLTMuonMatchAndPlot endRun()";

  // loop over all the histograms we booked, and handle the overflow bins

  // do this at end run, since you want to be sure you did it before you
  // saved your ME's.
  
  vector<MonitorElement*>::iterator iMonitorEl;
  
  for ( iMonitorEl = booked1DMonitorElements.begin();
        iMonitorEl != booked1DMonitorElements.end();
        iMonitorEl++ ) {

    moveOverflow((*iMonitorEl));

  }
  

}


void HLTMuonMatchAndPlot::finish()
{

  LogTrace ("HLTMuonVal") << "\n\nInside HLTMuonMatchAndPlot finish()" << endl;

  if (createStandAloneHistos && histoFileName != "") {
    dbe_->save(histoFileName);
  }
  
}


void HLTMuonMatchAndPlot::analyze( const Event & iEvent )
{

  LogTrace( "HLTMuonVal" ) << "\n\nIn analyze for trigger path " << 
    theTriggerName << ", Event:" << eventNumber <<"\n\n\n";

  
  // Call the selection method with the default selection
  bool validSelection = selectAndMatchMuons (iEvent, recMatches, hltFakeCands, mySelection);
  if (validSelection) fillPlots (recMatches, hltFakeCands);

  eventNumber++;
  
}// end analyze

bool HLTMuonMatchAndPlot::selectAndMatchMuons (const Event & iEvent, vector<MatchStruct> & myRecMatches,
                                               std::vector< std::vector<HltFakeStruct> > & myHltFakeCands
                                               ){

  LogTrace ("HLTMuonVal") << "\n\nInside selectAndMatchMuons, called with no selection argument"
                          << endl
                          << "Calling function using mySelection"
                          << endl;

  return selectAndMatchMuons (iEvent, myRecMatches, myHltFakeCands, mySelection);

}


bool HLTMuonMatchAndPlot::selectAndMatchMuons (const Event & iEvent, vector<MatchStruct> & myRecMatches,
                                               std::vector< std::vector<HltFakeStruct> > & myHltFakeCands,
                                               MuonSelectionStruct muonSelection){

  // clear the matches from the last event
  myRecMatches.clear();

  // a fake hlt cand is an hlt object not matched to a
  // reco object
  //  std::vector< std::vector<HltFakeStruct> > myHltFakeCands(numHltLabels);

  myHltFakeCands.clear();
  for (unsigned iLabel = 0; iLabel < numHltLabels; iLabel++){

    std::vector<HltFakeStruct> tempFake;
    
    myHltFakeCands.push_back(tempFake);
    
  }

  

  // Update event numbers
  // meNumberOfEvents->Fill(eventNumber);


  //------------------------------------------
  //    Trigger Requirement
  //    
  //------------------------------------------

  LogTrace("HLTMuonVal") << "Checking trigger result for "
                         << "trigger information stored in the following block "
                         << TriggerResultLabel;

  bool passedRequiredTrigger = applyTriggerSelection ( muonSelection, iEvent);

  if (!passedRequiredTrigger) {
    LogTrace ("HLTMuonVal") << "You didn't pass the required trigger"
                            << "skipping event"
                            << endl;
    return false;
  }

  //////////////////////////////////////////////////////////////////////////
  // Get all generated and reconstructed muons and create structs to hold  
  // matches to trigger candidates 

  double genMuonPt = -1;
  double recMuonPt = -1;


  LogTrace ("HLTMuonVal") << "\n\nStarting to look for gen muons\n\n";
                          
  
  //std::vector<MatchStruct> genMatches;
  

  LogTrace ("HLTMuonVal") << "\n\n\n\nDone getting gen, now getting reco\n\n\n";
  

  
  //std::vector<MatchStruct> highPtMatches;
  
  //reco::BeamSpot  beamSpot;
  foundBeamSpot = false;
  
  if ( useMuonFromReco ) {
    //Handle<reco::TrackCollection> muTracks;
    Handle<MuonCollection> muTracks;
    iEvent.getByLabel(RecoMuonInputTag, muTracks);    
    //reco::TrackCollection::const_iterator muon;
    MuonCollection::const_iterator muon;
    if  ( muTracks.failedToGet() ) {
      LogWarning("HLTMuonVal") << "WARNING: failed to get the RECO Muon collection named " << RecoMuonInputTag
                               << "\nYou have tracks to compare to... ignoring RECO muons"
                               << " for the rest of this job";
      useMuonFromReco = false;
    } else {

      LogTrace ("HLTMuonVal") << "Beginning loop over reco muons" << endl;
      
      for ( muon = muTracks->begin(); muon != muTracks->end(); ++muon ) {
        
        // this applies cuts that can
        // go towards the muon collection

        LogTrace ("HLTMuonVal") << "... Applying selection" << endl;
        if ( muonSelection.recoMuonSelector((*muon)) ) {

          // now apply cuts to the tracks.
          LogTrace ("HLTMuonVal") << "Passed selection!" << endl;
          
          if ( applyTrackSelection( muonSelection, (*muon) ) ){

            
          
            float pt  = muon->pt();
            float eta = muon->eta();
            MatchStruct newMatchStruct;
            newMatchStruct.recCand = &*muon;
            myRecMatches.push_back(newMatchStruct);

            LogTrace ("HLTMuonVal") << "\n\nFound a muon track in " << muonSelection.customLabel
                                    << " with pt = " << pt
                                    << ", eta = " << eta;
            // Take out this eta cut, but still check to see if
            // it is a new maximum pt
            //if ( pt > recMuonPt  && fabs(eta) < theMaxEtaCut)
            if (pt > recMuonPt )
              recMuonPt = pt;
            
          }
        }
      }
    }

    // This loop checks to see that we successfully stored our cands
    LogTrace ("HLTMuonVal") << "Print out all rec cands for " << muonSelection.customLabel
                            << endl;
    
    for (unsigned iMatch = 0; iMatch < myRecMatches.size(); iMatch++) {
      LogTrace ("HLTMuonVal") << "Cand #" << iMatch << "   ";
      LogTrace ("HLTMuonVal") << "Pt = " << myRecMatches[iMatch].recCand->pt()
                              << endl;
    }

    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByLabel(BeamSpotInputTag,recoBeamSpotHandle);
    if (!recoBeamSpotHandle.failedToGet()) {
      
      beamSpot = *recoBeamSpotHandle;
      foundBeamSpot = true;

      LogTrace ("HLTMuonVal") << "\n\n\nSUCESS finding beamspot\n\n\n" << endl;
      
    } else {
      LogWarning ("HLTMuonVal") << "FAILED to get the beamspot for this event";
    }
    

  } 
  
  LogTrace("HLTMuonVal") << "\n\n\n\ngenMuonPt: " << genMuonPt << ", "  
                         << "recMuonPt: " << recMuonPt
                         << "\nCustom name = " << muonSelection.customLabel << endl
                         << "\nNow preparing to get trigger objects" 
                         << "\n\n\n\n";

  //////////////////////////////////////////////////////////////////////////
  // Get the L1 and HLT trigger collections

  edm::Handle<trigger::TriggerEventWithRefs> rawTriggerEvent;
  edm::Handle<trigger::TriggerEvent>         aodTriggerEvent;
  vector<TriggerObject>                      l1Particles;
  vector<TriggerObject>                      l1RawParticles;
  vector<TriggerObject>                      l1Seeds;
  //--  HLTParticles [0] is a vector of L2 matches
  //--  HLTParticles [1] is a vector of L1 matches

  // HLT particles are just 4 vectors
  vector< vector<TriggerObject> >            hltParticles(numHltLabels);

  // HLT cands are references to trigger objects
  vector< vector<RecoChargedCandidateRef> >  hltCands(numHltLabels);

  // L1 Cands are references to trigger objects
  vector<L1MuonParticleRef> l1Cands;
  
  InputTag collectionTag;
  //size_t   filterIndex;


  // Try to get the triggerSummaryRAW branch for
  // this event. If it's there, great, keep using it.
  // but if it isn't there, skip over it silently

//   LogTrace ("HLTMuonVal") << "Trying to get RAW information\n\n";
                          
//   iEvent.getByLabel( HltRawInputTag, rawTriggerEvent );
  
//   if ( rawTriggerEvent.isValid() ) { 
//     LogTrace("HLTMuonVal") << "\n\nRAW trigger summary found! "
//                            << "\n\nUsing RAW information";
    
//     collectionTag = InputTag( theL1CollectionLabel, "", theHltProcessName );
//     filterIndex   = rawTriggerEvent->filterIndex(collectionTag);


//     if ( filterIndex < rawTriggerEvent->size() ) {
//       rawTriggerEvent->getObjects( filterIndex, TriggerL1Mu, l1Cands );
//       LogTrace ("HLTMuonVal") << "Found l1 raw cands for filter = " << filterIndex ;                              
        
//     } else {
//       LogTrace("HLTMuonVal") << "No L1 Collection with label " 
//                                 << collectionTag;
//     }
    
//     //for ( size_t i = 0; i < l1Cands.size(); i++ ) 
//     //  l1Cands.push_back( l1Cands[i]->p4() );
//     LogTrace ("HLTMuonVal") << "Looking for information from  hltFilters";
                            
//     for ( size_t i = 0; i < numHltLabels; i++ ) {

//       collectionTag = InputTag( theHltCollectionLabels[i], 
//                                 "", theHltProcessName );
//       filterIndex   = rawTriggerEvent->filterIndex(collectionTag);

//       LogTrace ("HLTMuonVal") << "Looking for candidates for filter "
//                               << theHltCollectionLabels[i]
//                               << ", index = "
//                               << filterIndex;
      
//       if ( filterIndex < rawTriggerEvent->size() )
//         rawTriggerEvent->getObjects( filterIndex, TriggerMuon, hltCands[i]);
//       else LogTrace("HLTMuonVal") << "No HLT Collection with label " 
//                                   << collectionTag;

//       // JMS -- do we ever store this raw info in the MatchStruct?
      

//       // don't copy the hltCands into particles
//       // for ( size_t j = 0; j < hltCands[i].size(); j++ )
//       // hltParticles[i].push_back( hltCands[i][j]->p4() );

//     } // End loop over theHltCollectionLabels
//   }  else {
//     LogTrace ("HLTMuonVal") << "\n\nCouldn't find any RAW information for this event";
                            
//   } // Done processing RAW summary information
    


  //// Get the candidates from the AOD trigger summary
  ///  JMS This is the unpacking that you might have
  ///  otherwise had to do 
  // if ( useAod ) {

    LogTrace ("HLTMuonVal") << "\n\n\nLooking for AOD branch named "
                            << "hltTriggerSummaryAOD\n\n\n";
                            
    iEvent.getByLabel(HltAodInputTag, aodTriggerEvent);
    if ( !aodTriggerEvent.isValid() ) { 
      LogInfo("HLTMuonVal") << "No AOD trigger summary found! Returning..."; 
      return false; 
    }

    LogTrace ("HLTMuonVal") << "\n\n\nFound a branch! Getting objects\n\n\n";


    // This gets you all of the stored trigger objects in the AOD block
    // could be muons, met, etc
    const TriggerObjectCollection objects = aodTriggerEvent->getObjects();

    LogTrace ("HLTMuonVal") << "\n\n\nFound a collection with size "
                            << objects.size() << "\n\n\n";

    if (objects.size() < 1) {
      LogTrace ("HLTMuonVal")
        << "You found the collection, but doesn't have any entries";

      return false;
    }

    // The AOD block has many collections, and you need to
    // parse which one you want. There are fancy lookup functions
    // to give you the number of the collection you want.
    // I think this is related to the trigger bits for each
    // event not being constant... so kinda like triger
    
    collectionTag = InputTag( theL1CollectionLabel, "", theHltProcessName );

    LogTrace ("HLTMuonVal") << "Trigger Name is " << theTriggerName;
    
    LogTrace ("HLTMuonVal") << "\n\n L1Collection tag is "
                            << collectionTag << endl
                            << " and size filters is "
                            << aodTriggerEvent->sizeFilters()
                            << "Looking up L1 information in trigSummaryAod";

    // this function call fills l1 particles with your matching trigger objects
    getAodTriggerObjectsForModule ( collectionTag, aodTriggerEvent, objects, l1Particles, muonSelection);
    
    int indexHltColl = 0;
    vector<string>::const_iterator iHltColl;
    for (iHltColl = theHltCollectionLabels.begin();
         iHltColl != theHltCollectionLabels.end();
         iHltColl++ ){
      collectionTag = InputTag((*iHltColl) , "", 
                               theHltProcessName );

      // this function call filles hltParticles with your hlt matches.
      getAodTriggerObjectsForModule ( collectionTag, aodTriggerEvent, objects, hltParticles[indexHltColl] , muonSelection);
      

      indexHltColl++;
    }


    // more very verbose debug
    // trying to restructure code 
    LogTrace ("HLTMuonVal") << "At the end of parsing the L2/L3 filters, you have found "
                            << "L2 = " <<  ((hltParticles.size() > 0) ? hltParticles[0].size() : 0)
                            << "L3 = " <<  ((hltParticles.size() > 1) ? hltParticles[1].size() : 0)
                            << endl;

    ///////////////////////////////////////////
    //
    //  Look up the L1 seeds
    //
    ///////////////////////////////////////////
    
    collectionTag = InputTag( theL1SeedModuleForHLTPath, "", theHltProcessName );

    LogTrace ("HLTMuonVal") << "\n\n L1Seed colelction tag is "
                            << collectionTag << endl
                            << " and size filters is "
                            << aodTriggerEvent->sizeFilters()
                            << "Looking up L1 Seed information in trigSummaryAod";

    // this function call fills l1 particles with your matching trigger objects
    getAodTriggerObjectsForModule ( collectionTag, aodTriggerEvent, objects, l1Seeds, muonSelection);
    
    LogTrace ("HLTMuonVal") << "At the end of parsing the L1 filter, you have found "
                            <<  l1Particles.size() << " objects: ";
  

  //////////////////////////////////////////////////////////////////////////
  // Initialize MatchStructs

  LorentzVector nullLorentzVector( 0., 0., 0., -999. );

  // trigger object id, pt
  TriggerObject nullTriggerObject (-9999, -9e10, -20, 0, 0);
  
  //L1MuonParticleRef nullL1Ref(L1MuonParticle(-1, nullLorentzVector));



  for ( size_t i = 0; i < myRecMatches.size(); i++ ) {
    myRecMatches[i].l1Cand = nullTriggerObject;
    myRecMatches[i].l1Seed = nullTriggerObject;
    myRecMatches[i].hltCands. assign( numHltLabels, nullTriggerObject );
    //myRecMatches[i].hltTracks.assign( numHltLabels, false );
    // new! raw matches too
    myRecMatches[i].hltRawCands.assign(numHltLabels, nullLorentzVector);
    myRecMatches[i].l1RawCand = nullLorentzVector;
  }




  
  //////////////////////////////////////////////////////////////////////////
  // Loop through L1 candidates, matching to gen/reco muons 

  numL1Cands = 0;

  
  for ( size_t i = 0; i < l1Particles.size(); i++ ) {

    TriggerObject l1Cand = l1Particles[i];
    double eta           = l1Cand.eta();
    double phi           = l1Cand.phi();
    // L1 pt is taken from a lookup table
    // double ptLUT      = l1Cand->pt();  

    double maxDeltaR = theL1DrCut;
    numL1Cands++;


    if ( useMuonFromReco ){
      int match = findRecMatch( eta, phi, maxDeltaR, myRecMatches );
      if ( match != -1 && myRecMatches[match].l1Cand.pt() < 0 ) {
        myRecMatches[match].l1Cand = l1Cand;
        LogTrace ("HLTMuonVal") << "Found a rec match to L1 particle (aod)  "
                                << " rec pt = " << myRecMatches[match].recCand->pt()
                                << ",  l1 pt  = " << myRecMatches[match].l1Cand.pt(); 
      } else {
        //hNumOrphansRec->getTH1F()->AddBinContent( 1 );
      }
    }

  } // End loop over l1Particles

  
  //========================================
  // Loop over L1 seeds and store matches
  //========================================
  
  for ( size_t i = 0; i < l1Seeds.size(); i++ ) {

    TriggerObject l1Cand = l1Seeds[i];
    double eta           = l1Cand.eta();
    double phi           = l1Cand.phi();
    // L1 pt is taken from a lookup table
    // double ptLUT      = l1Cand->pt();  

    double maxDeltaR = theL1DrCut;
    //numL1Cands++;


    if ( useMuonFromReco ){
      int match = findRecMatch( eta, phi, maxDeltaR, myRecMatches );
      if ( match != -1 && myRecMatches[match].l1Seed.pt() < 0 ) {
        myRecMatches[match].l1Seed = l1Cand;
        LogTrace ("HLTMuonVal") << "Found a rec match to L1 particle (aod)  "
                                << " rec pt = " << myRecMatches[match].recCand->pt()
                                << ",  l1 pt  = " << myRecMatches[match].l1Seed.pt(); 
      } else {
        //hNumOrphansRec->getTH1F()->AddBinContent( 1 );
      }
    }

  } // End loop over l1Seeds

  

  ////////////////////////////////////////////////////////
  //   Loop over the L1 Candidates (RAW information)
  //   and look for matches
  ////////////////////////////////////////////////////////
  
  for ( size_t i = 0; i < l1Cands.size(); i++ ) {

    LorentzVector l1Cand = l1Cands[i]->p4();
    
    double eta           = l1Cand.eta();
    double phi           = l1Cand.phi();
    // L1 pt is taken from a lookup table
    // double ptLUT      = l1Cand.pt();  

    double maxDeltaR = theL1DrCut;
    //numL1Cands++;


    if ( useMuonFromReco ){
      int match = findRecMatch( eta, phi, maxDeltaR, myRecMatches );
      if ( match != -1 && myRecMatches[match].l1RawCand.energy() < 0 ) {
        myRecMatches[match].l1RawCand = l1Cand;
        LogTrace ("HLTMuonVal") << "Found an L1 match to a RAW object";
      } else {
        //hNumOrphansRec->getTH1F()->AddBinContent( 1 );
      }
    }

  } // End loop over L1 Candidates (RAW)


  
  LogTrace("HLTMuonVal") << "Number of L1 Cands: " << numL1Cands;

  //////////////////////////////////////////////////////////////////////////
  // Loop through HLT candidates, matching to gen/reco muons

  vector<unsigned int> numHltCands( numHltLabels, 0) ;

  LogTrace ("HLTMuonVal") << "Looking for HLT matches for numHltLabels = "
                          << numHltLabels;
  
  for ( size_t i = 0; i < numHltLabels; i++ ) { 

    int triggerLevel      = ( i < ( numHltLabels / 2 ) ) ? 2 : 3;
    double maxDeltaR      = ( triggerLevel == 2 ) ? theL2DrCut : theL3DrCut;

    LogTrace ("HLTMuonVal") << "Looking at 4-vectors  for " << theHltCollectionLabels[i];
    
    for ( size_t candNum = 0; candNum < hltParticles[i].size(); candNum++ ) {

      TriggerObject hltCand = hltParticles[i][candNum];
      double eta            = hltCand.eta();
      double phi            = hltCand.phi();

      numHltCands[i]++;


      if ( useMuonFromReco ){

        HltFakeStruct tempFakeCand; 
        tempFakeCand.myHltCand  = hltCand;

        int match  = findRecMatch( eta, phi, maxDeltaR, myRecMatches );

        // if match doesn't return error (-1)
        // and if this candidate spot isn't filled
        if ( match != -1 && myRecMatches[match].hltCands[i].pt() < 0 ) {
          myRecMatches[match].hltCands[i] = hltCand;

          LogTrace ("HLTMuonVal") << "Found a HLT cand match!   "
                                  << " rec pt = " << myRecMatches[match].recCand->pt()
                                  << ",   hlt pt = " << myRecMatches[match].hltCands[i].pt();

          // since this matched, it's not a fake, so
          // record it as "not a fake"
          tempFakeCand.isAFake = false;

          
          // if match *did* return -1, then this is a fake  hlt candidate
          // it is fake because it isn't matched to a reco muon
          // 2009-03-24 oops, found a bug here, used to be != -1
          // fixed 
        } else if (match == -1){
          tempFakeCand.isAFake = true;
          //hNumOrphansRec->getTH1F()->AddBinContent( i + 2 );
        }

        // add this cand 
        myHltFakeCands[i].push_back(tempFakeCand);
        LogTrace ("HLTMuonVal") << "\n\nWas this a fake hlt cand? "
                              << tempFakeCand.isAFake;

      }

                              
      
      LogTrace("HLTMuonVal") << "Number of HLT Cands: " << numHltCands[i];

    } // End loop over HLT particles

    
    LogTrace ("HLTMuonVal") << "Looking at RAW Candidates for "
                            << theHltCollectionLabels[i];

    
    for ( size_t candNum = 0; candNum < hltCands[i].size(); candNum++ ) {

      LorentzVector hltCand = hltCands[i][candNum]->p4();
      double eta            = hltCand.eta();
      double phi            = hltCand.phi();

      numHltCands[i]++;


      if ( useMuonFromReco ){

        //HltFakeStruct tempFakeCand; 
        //tempFakeCand.myHltCand  = hltCand;

        int match  = findRecMatch( eta, phi, maxDeltaR, myRecMatches );

        // if match doesn't return error (-1)
        // and if this candidate spot isn't filled
        if ( match != -1 && myRecMatches[match].hltCands[i].pt() < 0 ) {
          myRecMatches[match].hltRawCands[i] = hltCand;
          LogTrace ("HLTMuonVal") << "Found a RAW hlt match to reco";
        }

        //else if (match == -1){
          //tempFakeCand.isAFake = true;
          //hNumOrphansRec->getTH1F()->AddBinContent( i + 2 );
          //}

        // add this cand 
        //myHltFakeCands[i].push_back(tempFakeCand);
        //LogTrace ("HLTMuonVal") << "\n\nWas this a fake hlt cand? "
        //                      << tempFakeCand.isAFake;

      }

                              
      
      //LogTrace("HLTMuonVal") << "Number of HLT Cands: " << numHltCands[i];

    } // End loop over HLT RAW information


  } // End loop over HLT labels


  // if you reach this point, then the code was
  // successful

  return true;
  
}// end select and match muons


void HLTMuonMatchAndPlot::fillPlots (vector<MatchStruct> & myRecMatches,
                                     std::vector< std::vector<HltFakeStruct> > & myHltFakeCands) {


  if (!dbe_) {

    LogTrace ("HLTMuonVal")
      << "===Warning=== You've tried to call fill plots, "
      << "but no DQMStore object exists... refusing to fill plots"
      << endl;

    return;

  }

  int numRecMatches = myRecMatches.size();

  

  //double recMuonPt = -1;
  
  //=======================
  // DoubleMu Triggers
  // ----------------------
  // If you're using a double mu trigger
  // Check to see if you found at least two reco muons
  // If you haven't, then skip this event!
  //========================

  if ((theNumberOfObjects == 2) && (myRecMatches.size() < 2)) return;
  
  //////////////////////////////////////////////////////////////////////////
  // Fill histograms
  
  ////////////////////////////////////////////
  //
  //               RECO Matching
  //
  ///////////////////////////////////////////

  double maxMatchPtRec = -10.0;
  //std::vector <double> allRecPts;
  //std::vector <bool> matchedToHLT;
  
  // Look at each rec & hlt cand

  for ( size_t i = 0; i < myRecMatches.size(); i++ ) {

    LogTrace("HLTMuonVal") << "Reco Candidate loop:"
                           << "looking at cand " << i
                           << " out of " << myRecMatches.size()
                           << endl;


    if ((isL3Path || isL2Path) && requireL1SeedForHLTPaths) {

      LogTrace ("HLTMuonVal") << "Checking to see if your RECO muon matched to an L1 seed"
                              << endl;

      if (myRecMatches[i].l1Seed.pt() < 0) {
        LogTrace ("HLTMuonVal") << "No match to L1 seed, skipping this RECO muon" << endl;
        continue;
      }
    }

    

    double pt  = myRecMatches[i].recCand->pt();
    double eta = myRecMatches[i].recCand->eta();
    double phi = myRecMatches[i].recCand->phi();
    int recPdgId = myRecMatches[i].recCand->pdgId();

    LogTrace ("HLTMuonVal") << "trying to get a global track for this muon" << endl;

    // old way - breaks if no global track
    //TrackRef theMuoGlobalTrack = myRecMatches[i].recCand->globalTrack();

    TrackRef theMuonTrack = getCandTrackRef (mySelection, (*myRecMatches[i].recCand));
    
    double d0 = -9e20;
    double z0 = -9e20;
    int charge = -99999;
    int plottedCharge = -99999;

    double d0beam = -9e20;
    double z0beam = -9e20;
    
    if (theMuonTrack.isNonnull() ) {
      d0 = theMuonTrack->d0();
      z0 = theMuonTrack->dz();
      // comment:
      // does the charge function return the
      // same value as the abs(pdgId) ?    
      charge = theMuonTrack->charge(); 
      plottedCharge = getCharge (recPdgId);
      
    
      if (foundBeamSpot) {
        d0beam = theMuonTrack->dxy(beamSpot.position());
        z0beam = theMuonTrack->dz(beamSpot.position());
        
        hBeamSpotZ0Rec[0]->Fill(beamSpot.z0());
      }


    } else {
      LogTrace ("HLTMuonVal") << "... oops! that wasn't a global muon" << endl;
    }
    
    
    // For now, take out the cuts on the pt/eta,
    // We'll get the total efficiency and worry about
    // the hlt matching later.    
    //    if ( pt > theMinPtCut &&  fabs(eta) < theMaxEtaCut ) {
    
    //hNumObjects->getTH1()->AddBinContent(2);

    // fill the "all" histograms for basic muon
    // parameters
    hPassEtaRec[0]->Fill(eta);
    hPassPhiRec[0]->Fill(phi);
    hPassPtRec[0]->Fill(pt);
    hPhiVsEtaRec[0]->Fill(eta,phi);
    hPassD0Rec[0]->Fill(d0);
    hPassD0BeamRec[0]->Fill(d0beam);
    hPassZ0Rec[0]->Fill(z0);
    hPassZ0BeamRec[0]->Fill(z0beam);
    hPassCharge[0]->Fill(charge);
    
    MuonIsolation thisIso = myRecMatches[i].recCand->isolationR03();
    double emEnergy = thisIso.emEt;
    double hadEnergy = thisIso.hadEt;
    double myMuonIso = (emEnergy + hadEnergy) / pt;

    hIsolationRec[0]->Fill(myMuonIso);
    
    if (numRecMatches == 1) {
      hPassPtRecExactlyOne[0]->Fill(pt);
    }
    

    // if you found an L1 match, fill this histo
    // check for L1 match using pt, not energy
    if ( (myRecMatches[i].l1Cand.pt() > 0) && ((useFullDebugInformation) || (isL1Path)) ) {
      hPassEtaRec[1]->Fill(eta);
      hPassPhiRec[1]->Fill(phi);
      hPassPtRec[1]->Fill(pt);
      hPhiVsEtaRec[1]->Fill(eta,phi);
      hPassD0Rec[1]->Fill(d0);
      hPassD0BeamRec[1]->Fill(d0beam);
      hPassZ0Rec[1]->Fill(z0);
      hPassZ0BeamRec[1]->Fill(z0beam);
      hPassCharge[1]->Fill(charge);
      hIsolationRec[1]->Fill(myMuonIso);

      double l1eta = myRecMatches[i].l1Cand.eta();
      double l1phi = myRecMatches[i].l1Cand.phi();
      double l1pt  = myRecMatches[i].l1Cand.energy();

      // Get the charges in terms of charge constants
      // this reduces bins in histogram.
      int l1plottedCharge = getCharge (myRecMatches[i].l1Cand.id());
      LogTrace ("HLTMuonVal") << "The pdg id is (L1)   "
                              << myRecMatches[i].l1Cand.id()
                              << "  and the L1 plotted charge is "
                              << l1plottedCharge;
      
      
      double deltaR = reco::deltaR (l1eta, l1phi, eta, phi);

      double deltaPhi = reco::deltaPhi (l1phi, phi);
      
      // These are matched histos
      // so they have no "all" histos
      //
      
      hDeltaRMatched[0]->Fill(deltaR);
      hPassMatchPtRec[0]->Fill(pt);
      //hPtMatchVsPtRec[0]->Fill(l1pt, pt);
      //hEtaMatchVsEtaRec[0]->Fill(l1eta, eta);
      //hPhiMatchVsPhiRec[0]->Fill(l1phi, phi);
      hMatchedDeltaPhi[0]->Fill(deltaPhi);
      //hDeltaPhiVsPhi[0]->Fill(phi, deltaPhi);
      //hDeltaPhiVsZ0[0]->Fill(z0, deltaPhi);
      //hDeltaPhiVsD0[0]->Fill(d0, deltaPhi);
      // Resolution histos must have hlt matches
      
      hResoPtAodRec[0]->Fill((pt - l1pt)/pt);
      hResoEtaAodRec[0]->Fill((eta - l1eta)/fabs(eta));
      hResoPhiAodRec[0]->Fill((phi - l1phi)/fabs(phi));
        
      
      hChargeFlipMatched[0]->Fill(l1plottedCharge, plottedCharge);
      
      if (numRecMatches == 1) {
        hPassExaclyOneMuonMaxPtRec[1]->Fill(pt);
        hPassPtRecExactlyOne[1]->Fill(pt);
      }
    }
    
    //  bool foundAllPreviousCands = true;
    //  Look through the hltCands and see what's going on
    //

    
    for ( size_t j = 0; j < myRecMatches[i].hltCands.size(); j++ ) {
      if ( myRecMatches[i].hltCands[j].pt() > 0 ) {
        double hltCand_pt = myRecMatches[i].hltCands[j].pt();
        double hltCand_eta = myRecMatches[i].hltCands[j].eta();
        double hltCand_phi = myRecMatches[i].hltCands[j].phi();
        int hltCand_plottedCharge = getCharge(myRecMatches[i].hltCands[j].id());

        // store this rec muon pt, not hlt cand pt
        if (theHltCollectionLabels.size() > j) {
          TString tempString = theHltCollectionLabels[j];
          if (tempString.Contains("L3")) {
            
            maxMatchPtRec = (pt > maxMatchPtRec)? pt : maxMatchPtRec;
          }
        }

        // these are histos where you have
        // all + L1 (= displaced two indices)
        // Which means your HLT histos are
        // at index j+HLT_PLOT_OFFSET 
        hPassEtaRec[j+HLT_PLOT_OFFSET]->Fill(eta);
        hPassPhiRec[j+HLT_PLOT_OFFSET]->Fill(phi);
        hPassPtRec[j+HLT_PLOT_OFFSET]->Fill(pt);
        hPhiVsEtaRec[j+HLT_PLOT_OFFSET]->Fill(eta,phi);
        hPassD0Rec[j+HLT_PLOT_OFFSET]->Fill(d0);
        hPassD0BeamRec[j+HLT_PLOT_OFFSET]->Fill(d0beam);
        hPassZ0Rec[j+HLT_PLOT_OFFSET]->Fill(z0);
        hPassZ0BeamRec[j+HLT_PLOT_OFFSET]->Fill(z0beam);
        hPassCharge[j+HLT_PLOT_OFFSET]->Fill(charge);
        hIsolationRec[j+HLT_PLOT_OFFSET]->Fill(myMuonIso);
        
        
        // Histograms with Match in the name only have HLT
        // matches possible
        // so there are no "all" histograms
        // so offset = 1 b/c of L1 histos

        double deltaR = reco::deltaR (hltCand_eta, hltCand_phi,
                                        eta, phi);

        double deltaPhi = reco::deltaPhi (hltCand_phi, phi);

        hDeltaRMatched[j+HLT_PLOT_OFFSET-1]->Fill(deltaR);
        hPassMatchPtRec[j+HLT_PLOT_OFFSET-1]->Fill(pt);
        //hPtMatchVsPtRec[j+HLT_PLOT_OFFSET-1]->Fill(hltCand_pt, pt);
        //hEtaMatchVsEtaRec[j+HLT_PLOT_OFFSET-1]->Fill(hltCand_eta, eta);
        //hPhiMatchVsPhiRec[j+HLT_PLOT_OFFSET-1]->Fill(hltCand_phi, phi);
        hMatchedDeltaPhi[j+HLT_PLOT_OFFSET-1]->Fill(deltaPhi);
        //hDeltaPhiVsPhi[j+HLT_PLOT_OFFSET-1]->Fill(phi, deltaPhi);
        //hDeltaPhiVsZ0[j+HLT_PLOT_OFFSET-1]->Fill(z0, deltaPhi);
        //hDeltaPhiVsD0[j+HLT_PLOT_OFFSET-1]->Fill(d0, deltaPhi);
        

        LogTrace ("HLTMuonVal") << "The pdg id is (hlt [" << j << "]) "
                                << myRecMatches[i].hltCands[j].id()
                                << "  and the plotted charge is "
                                << hltCand_plottedCharge
                                << ", w/ rec  charge "
                                << charge
                                << ", and plotted charge "
                                << plottedCharge
                                << "\n                "
                                << "and rec pdg id = "
                                << recPdgId;
        

        
        hChargeFlipMatched[j+HLT_PLOT_OFFSET-1]->Fill( hltCand_plottedCharge, plottedCharge);

        
        // Resolution histos must have hlt matches

        hResoPtAodRec[j+HLT_PLOT_OFFSET-1]->Fill((pt - hltCand_pt)/pt);
        hResoEtaAodRec[j+HLT_PLOT_OFFSET-1]->Fill((eta - hltCand_eta)/fabs(eta));
        hResoPhiAodRec[j+HLT_PLOT_OFFSET-1]->Fill((phi - hltCand_phi)/fabs(phi));
        
        if (numRecMatches == 1 && (myRecMatches[i].hltCands.size()== 1)) {
          hPassExaclyOneMuonMaxPtRec[j+HLT_PLOT_OFFSET]->Fill(pt);
          hPassPtRecExactlyOne[j+HLT_PLOT_OFFSET]->Fill(pt);
        }
      } // end if found hlt match      
    }

    /////////////////////////////////////////////////
    //         Fill some RAW histograms
    /////////////////////////////////////////////////
    if (useFullDebugInformation) {
      LogTrace ("HLTMuonVal")  << "\n.... now Filling Raw Histos";
      if ( myRecMatches[i].l1RawCand.energy() > 0 ) {
      
        // you've found a L1 raw candidate
        rawMatchHltCandPt[1]->Fill(pt);
        rawMatchHltCandEta[1]->Fill(eta);
        rawMatchHltCandPhi[1]->Fill(phi);      
      }

      LogTrace ("HLTMuonVal") << "There are " << myRecMatches[i].hltCands.size()
                              << " hltRaw candidates that could match, starting loop"
                              << endl;
    
      for ( size_t j = 0; j < myRecMatches[i].hltCands.size(); j++ ) {
        if ( myRecMatches[i].hltCands[j].pt() > 0 ) {
          rawMatchHltCandPt[j+HLT_PLOT_OFFSET]->Fill(pt);
          rawMatchHltCandEta[j+HLT_PLOT_OFFSET]->Fill(eta);
          rawMatchHltCandPhi[j+HLT_PLOT_OFFSET]->Fill(phi);   
        }
      }

    }
  } // end RECO matching

  /////////////////////////////////////////
  //
  //  HLT fakes cands
  // 
  /////////////////////////////////////////

  LogTrace ("HLTMuonVal")  << "\n.... now looping over fake cands";
  for (unsigned int  iHltModule = 0;  iHltModule < numHltLabels; iHltModule++) {
    for(size_t iCand = 0; iCand < myHltFakeCands[iHltModule].size() ; iCand ++){
      LogTrace ("HLTMuonVal") << "Label number : " << iHltModule
                              << "(max = " << numHltLabels << ")\n"
                              << "Candidate number: " << iCand
                              << "(max = " <<  myHltFakeCands[iHltModule].size()
                              << " )\n";
        
                              
      TriggerObject candVect = myHltFakeCands[iHltModule][iCand].myHltCand;
      bool candIsFake = myHltFakeCands[iHltModule][iCand].isAFake;
      
      allHltCandPt[iHltModule]->Fill(candVect.pt());
      allHltCandEta[iHltModule]->Fill(candVect.eta());
      allHltCandPhi[iHltModule]->Fill(candVect.phi());

      if (candIsFake) {
        fakeHltCandPt[iHltModule]->Fill(candVect.pt());
        fakeHltCandEta[iHltModule]->Fill(candVect.eta());
        fakeHltCandPhi[iHltModule]->Fill(candVect.phi());
        //fakeHltCandEtaPhi[iHltModule]->Fill(candVect.eta(), candVect.phi());

        // JMS extra hack - print out run,event so you can look
        // in event display
        // int myRun = iEvent.id().run();
        //         int myEvent = iEvent.id().event();
        

        //         cout << endl << "FAKE! run = " << myRun << ", event = "
        //              << myEvent << ", pt = " << candVect.pt() << ", eta = "
        //              << candVect.eta() << "phi, " << candVect.phi() << endl << endl;
        
      }
      
    }
    
  }
  

  LogTrace ("HLTMuonVal") << "There are " << myRecMatches.size()
                          << "  RECO muons in this event"
                          << endl;
    
  LogTrace ("HLTMuonVal") << "The max pt found by looking at candiates is   "
                          << maxMatchPtRec
    //<< "\n and the max found while storing reco was "
    //<< recMuonPt
                          << endl;
  
  ///////////////////////////////////////////
  //
  //  Fill MAX PT plot
  //
  ////////////////////////////////////////////


  // genMuonPt and maxMatchPtRec are the max values
  // fill these hists with the max reconstructed Pt  
  //if ( genMuonPt > 0 ) hPassMaxPtGen[0]->Fill( genMuonPt );
  if ( maxMatchPtRec > 0 ) hPassMaxPtRec[0]->Fill( maxMatchPtRec );

  // there will be one hlt match for each
  // trigger module label
  // int numHltMatches = myRecMatches[i].hltCands.size();

  if (numRecMatches == 1) {
    if (maxMatchPtRec >0) hPassExaclyOneMuonMaxPtRec[0]->Fill(maxMatchPtRec);
  }

  // Fill these if there are any L1 candidates
  if (useFullDebugInformation || isL1Path) {
    if ( numL1Cands >= theNumberOfObjects ) {
      //if ( genMuonPt > 0 ) hPassMaxPtGen[1]->Fill( genMuonPt );
      if ( maxMatchPtRec > 0 ) hPassMaxPtRec[1]->Fill( maxMatchPtRec );
      if (numRecMatches == 1 && numL1Cands == 1) {
        if (maxMatchPtRec >0) hPassExaclyOneMuonMaxPtRec[1]->Fill(maxMatchPtRec);
      }
    }
  }


  
  for ( size_t i = 0; i < numHltLabels; i++ ) {
    // this will only fill up if L3
    // I don't think it's correct to fill
    // all the labels with this
    if (maxMatchPtRec > 0) hPassMaxPtRec[i+HLT_PLOT_OFFSET]->Fill(maxMatchPtRec);

  }


  

} // end fillPlots: Done filling histograms



const reco::Candidate* HLTMuonMatchAndPlot::
findMother( const reco::Candidate* p ) 
{
  const reco::Candidate* mother = p->mother();
  if ( mother ) {
    if ( mother->pdgId() == p->pdgId() ) return findMother(mother);
    else return mother;
  }
  else return 0;
}


////////////////////////  WARNING   ///////////////////////////////
//
//      You should not use findGenMatch b/c it references sim info
//////////////////////////////////////////////////////////////////

int HLTMuonMatchAndPlot::findGenMatch
( double eta, double phi, double maxDeltaR, vector<MatchStruct> matches )
{
  double bestDeltaR = maxDeltaR;
  int bestMatch = -1;
  for ( size_t i = 0; i < matches.size(); i++ ) {
    // double dR = reco::deltaR( eta, phi, 
    // 				matches[i].genCand->eta(), 
    // 				matches[i].genCand->phi() );

    
    double dR = 10;
    
    if ( dR  < bestDeltaR ) {
      bestMatch  =  i;
      bestDeltaR = dR;
    }
  }
  return bestMatch;
}



int HLTMuonMatchAndPlot::findRecMatch
( double eta, double phi,  double maxDeltaR, vector<MatchStruct> matches)
{
  double bestDeltaR = maxDeltaR;
  int bestMatch = -1;

  // Case for delta R matching
  // the != cosmic case is for default handling.
  if (matchType != "cosmic" || matchType == "dr"  ) {
    for ( size_t i = 0; i < matches.size(); i++ ) {
      double dR = reco::deltaR( eta, phi, 
                                  matches[i].recCand->eta(), 
                                  matches[i].recCand->phi() );
      if ( dR  < bestDeltaR ) {
        bestMatch  =  i;
        bestDeltaR = dR;
      }
    }
    return bestMatch;
  }

  if (matchType == "cosmic") {

    //   Comsic trigger matching
    //   Just require the the muon
    //   will be in the same half of the detector
    //   ignore the eta information
    //   but we will look for the minmum delta phi
    //   with the muon in that region of the detector
    
    double bestDphi = 100.0;
    for ( size_t i = 0; i < matches.size(); i++ ) {

      double recCandPhi = matches[i].recCand->phi();

      
      if (recCandPhi < 0 && phi < 0) {
        if ( reco::deltaPhi(phi, recCandPhi) < bestDphi) {
          bestDphi = reco::deltaPhi(phi, recCandPhi);
          bestMatch = i;          
        }
      }

     
      if (recCandPhi > 0 && phi > 0) {
        
        if ( reco::deltaPhi(phi, recCandPhi) < bestDphi) {
          bestDphi = reco::deltaPhi(phi, recCandPhi);
          bestMatch = i;          
        }         
          
      }        
      
    }
    return bestMatch;
  }

  // If you get here, then you've improperly set
  // your matching

  LogWarning ("HLTMuonVal") << "WARNING: You have improperly set matchType" << endl
                         << "valid choices are 'dr' and 'cosmic', " <<endl
                         << "but you provided    " << matchType << endl;
  
  return bestMatch;
  
}


bool HLTMuonMatchAndPlot::applyTrackSelection (MuonSelectionStruct mySelection, Muon candMuon) {

  LogTrace ("HLTMuonVal") << "Applying track selection to your muon"
                          << endl;
  // get the track 
  // you should have specified the track using the collection names
  TrackRef theMuonTrack = getCandTrackRef (mySelection, candMuon);

  bool passedSelection = false;
  
  if ( theMuonTrack.isNonnull() ) {
     double d0 = theMuonTrack->d0();
     double z0 = theMuonTrack->dz();


     LogTrace ("HLTMuonVal") << "d0 = " << d0
                             << ", d0 cut = " << mySelection.d0cut << endl
                             << "z0 = " << z0
                             << ", z0 cut = " << mySelection.z0cut << endl;
                             
                             
     
     if (fabs(d0) < mySelection.d0cut &&
         fabs(z0) < mySelection.z0cut ) {
       passedSelection = true;
     }
  } else {
    LogTrace ("HLTMuonVal") << "This event didn't have a valid track of type "
                            << mySelection.trackCollection;            
  }

  return passedSelection;
  
}

bool HLTMuonMatchAndPlot::applyTriggerSelection(MuonSelectionStruct mySelection, const Event & event) {

  bool passedAnyTrigger = false;
  //  Look and your event selection criteria
  //  if you have a vector of size zero
  //  or if you have a vector with just an empty string
  //  then you should just skip this selection and return true

  LogTrace ("HLTMuonVal") << "Checking to see if you have non-empty triggers to match"
                          << endl;
  
  if (mySelection.requiredTriggers.size() < 1)
    return true;

  vector<string>::const_iterator iTargetTrig;

  bool everythingIsEmpty = true;
  for ( iTargetTrig = mySelection.requiredTriggers.begin();
        iTargetTrig != mySelection.requiredTriggers.end();
        iTargetTrig ++ ) {

    if ( (*iTargetTrig) != "" ) {
      everythingIsEmpty = false;
    }
    
  }

  if (everythingIsEmpty) {
    LogTrace ("HLTMuonVal") << "Only empty triggers, skipping match";
    return true;
  }

  //  At this point, you have a true trigger requirement
  //  You need to check the trigger results
  //  0. Get the trigger resutls 
  //  1. Loop over list of target triggers 
  //  2. See if the target is valid according to HLTConfig
  //  3. If it is, check to see that it fired
  //
  //  Potential optimization - store the trigger index
  //  rather than doing the match for each event


  // Get the trigger results
  
  Handle<TriggerResults> trigRes;
  event.getByLabel(TriggerResultLabel, trigRes);
  if (!trigRes.isValid()){
    edm::InputTag triggerResultsLabelFU(TriggerResultLabel.label(),TriggerResultLabel.instance(), "FU");
    event.getByLabel(triggerResultsLabelFU,trigRes);
    if(!trigRes.isValid()) {
      LogTrace("HLTMuonVal")<< "Trigger Results WARNING: No trigger Results in event info, but you wanted to check a trigger";
      // Do nothing, and 
      //TrigResultsIn=false;

      return false;
    }
  }
  unsigned size = trigRes->size();

  unsigned int Ntp = 0;
  
  LogTrace("HLTMuonVal")<< "Ntp=" << Ntp <<" Size of trigger results="<<size;


  // loop over the list of target triggers  
  
  map<string,bool> firedTrigger;
  
  for ( iTargetTrig = mySelection.requiredTriggers.begin();
        iTargetTrig != mySelection.requiredTriggers.end();
        iTargetTrig++ ) {

    std::string targetName = (*iTargetTrig);

    LogTrace("HLTMuonVal") << "Looking to see if " << targetName << " has fired... ";

    firedTrigger[targetName] = false;
    vector<string>::const_iterator iValidTrig;
    unsigned int trigIndex = 0;
    for ( iValidTrig = selectedValidTriggers.begin();
          iValidTrig != selectedValidTriggers.end();
          iValidTrig ++) {

      if ( targetName == (*iValidTrig)){
        
        LogTrace ("HLTMuonVal") << "Trigger " << targetName
                                << " was part of the hlt configuration at index"
                                << trigIndex
                                << endl;
        
        firedTrigger[targetName] =  trigRes->accept(trigIndex);

        LogTrace ("HLTMuonVal") << "Did the trigger fire?      "
                                << ((firedTrigger[targetName]) ? "PASSED" : "FAILED")
                                << endl;
        
      }

      trigIndex++;
    } // end loop over valid triggers
  }// end loop over target triggers
    

  map<string,bool>::const_iterator iResult;

  passedAnyTrigger = false;

  LogTrace ("HLTMuonVal") << "OR-ing trigger results together" <<endl;

  
  for (iResult = firedTrigger.begin();
       iResult != firedTrigger.end();
       iResult ++) {

    passedAnyTrigger = passedAnyTrigger || iResult->second;
    
  }

  LogTrace ("HLTMuonVal") << "Returning " << passedAnyTrigger;

  return passedAnyTrigger;
  
}



TrackRef HLTMuonMatchAndPlot::getCandTrackRef (MuonSelectionStruct mySelection, Muon candMuon) {

  string trackCollection = mySelection.trackCollection;
  TrackRef theTrack;

  LogTrace ("HLTMuonVal") << "Getting the track reference for coll "
                          << trackCollection
                          << endl;

  LogTrace ("HLTMuonVal") << "Muon information" << endl
                          << "pt = " << candMuon.pt()
                          << ", phi = " << candMuon.phi()
                          << ", eta = " << candMuon.eta()
                          << ", global muon? = " << candMuon.isGlobalMuon()
                          << ", standalone muon = " << candMuon.isStandAloneMuon()
                          << ", tracker muon = " << candMuon.isTrackerMuon()
                          << endl;
  
  if (trackCollection == "innerTrack") {
    LogTrace ("HLTMuonVal") << "----> GET " << trackCollection;

    if ( candMuon.isTrackerMuon() ) {
      theTrack = candMuon.innerTrack();
    }
    
  } else if ( trackCollection == "outerTrack" ) {
    
    LogTrace ("HLTMuonVal") << "----> GET " << trackCollection;

    if ( candMuon.isStandAloneMuon() ) {
      theTrack = candMuon.outerTrack();
    }
    
  } else if ( trackCollection == "globalTrack") {

    LogTrace ("HLTMuonVal") << "----> GET " << trackCollection;
    if (candMuon.isGlobalMuon()) {
      theTrack = candMuon.globalTrack();
    }
  }

  if (theTrack.isNonnull()) {
    LogTrace ("HLTMuonVal") << "Found the desired track";
  } else {
    LogTrace ("HLTMuonVal") << "No track for this candidate";
  }
  
  return theTrack;
}


void HLTMuonMatchAndPlot::begin() 
{
  LogTrace ("HLTMuonVal") << "\n\nInside HLTMuonMatchAndPlot begin()";

  TString myLabel, newFolder;
  vector<TH1F*> h;

  if ( dbe_ ) {
    dbe_->cd();
    dbe_->setCurrentFolder("HLT/Muon");

    // JMS I think this is trimming all L1 names to
    // to be L1Filtered
    // Update this so that it really is just L1Filtered
    // new trigger names ruin string trimming

    if (useOldLabels) { 
      myLabel = theL1CollectionLabel;
      myLabel = myLabel(myLabel.Index("L1"),myLabel.Length());
      myLabel = myLabel(0,myLabel.Index("Filtered")+8);
    } else {
      myLabel = "L1Filtered";
    }
    // JMS Old way of doing things
    //newFolder = "HLT/Muon/Distributions/" + theTriggerName;
    newFolder = "HLT/Muon/Distributions/" + theTriggerName + "/" + mySelection.customLabel;

    
    
    dbe_->setCurrentFolder( newFolder.Data() );

    //meNumberOfEvents            = dbe_->bookInt("NumberOfEvents");
    //MonitorElement *meMinPtCut  = dbe_->bookFloat("MinPtCut"    );
    //MonitorElement *meMaxEtaCut = dbe_->bookFloat("MaxEtaCut"   );
    //meMinPtCut ->Fill(theMinPtCut );
    //meMaxEtaCut->Fill(theMaxEtaCut);
    
    vector<string> binLabels;
    binLabels.push_back( theL1CollectionLabel.c_str() );
    for ( size_t i = 0; i < theHltCollectionLabels.size(); i++ )
      binLabels.push_back( theHltCollectionLabels[i].c_str() );

    //hNumObjects = dbe_->book1D( "numObjects", "Number of Objects", 7, 0, 7 );
    //hNumObjects->setBinLabel( 1, "Gen" );
    //hNumObjects->setBinLabel( 2, "Reco" );
    //for ( size_t i = 0; i < binLabels.size(); i++ )
    //hNumObjects->setBinLabel( i + 3, binLabels[i].c_str() );
    //hNumObjects->getTH1()->LabelsDeflate("X");


    if ( useMuonFromReco ){

      //hNumOrphansRec = dbe_->book1D( "recNumOrphans", "Number of Orphans;;Number of Objects Not Matched to a Reconstructed #mu", 5, 0, 5 );
      //       for ( size_t i = 0; i < binLabels.size(); i++ )
      //         hNumOrphansRec->setBinLabel( i + 1, binLabels[i].c_str() );
      //hNumOrphansRec->getTH1()->LabelsDeflate("X");


      
      
      // 0 = MaxPt_All
      hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_All", "pt of Leading Reco Muon" ,  numBinsInPtHisto, ptBins) );
      // 1 = MaxPt if matched to L1 Trigger
      if (useFullDebugInformation || isL1Path) hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_" + myLabel, "pt of Leading Reco Muon, if matched to " + myLabel,  numBinsInPtHisto, ptBins) );

      hPassEtaRec.push_back( bookIt( "recPassEta_All", "#eta of Reco Muons", theEtaParameters) );
      if (useFullDebugInformation || isL1Path) hPassEtaRec.push_back( bookIt( "recPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
      
      hPassPhiRec.push_back( bookIt( "recPassPhi_All", "#phi of Reco Muons", thePhiParameters) );
      if (useFullDebugInformation || isL1Path) hPassPhiRec.push_back( bookIt( "recPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );
      

      
      hPassPtRec.push_back( bookIt( "recPassPt_All", "Pt of  Reco Muon" , numBinsInPtHisto, ptBins) );
      if (useFullDebugInformation || isL1Path) hPassPtRec.push_back( bookIt( "recPassPt_" + myLabel, "pt  Reco Muon, if matched to " + myLabel,  numBinsInPtHisto, ptBins) );
      
      hPassPtRecExactlyOne.push_back( bookIt( "recPassPtExactlyOne_All", "pt of Leading Reco Muon (==1 muon)" ,  numBinsInPtHisto, ptBins) );
      if (useFullDebugInformation || isL1Path) hPassPtRecExactlyOne.push_back( bookIt( "recPassPtExactlyOne_" + myLabel, "pt of Leading Reco Muon (==1 muon), if matched to " + myLabel,  numBinsInPtHisto, ptBins) );
      
      hPassExaclyOneMuonMaxPtRec.push_back( bookIt("recPassExactlyOneMuonMaxPt_All", "pt of Leading Reco Muon in events with exactly one muon" ,  numBinsInPtHisto, ptBins) );
      if (useFullDebugInformation || isL1Path) hPassExaclyOneMuonMaxPtRec.push_back( bookIt("recPassExactlyOneMuonMaxPt_" + myLabel, "pt of Leading Reco Muon in events with exactly one muon match to " + myLabel ,  numBinsInPtHisto, ptBins) );

      hPassD0Rec.push_back( bookIt("recPassD0_All", "Track 2-D impact parameter wrt (0,0,0)(d0) ALL", theD0Parameters));
      if (useFullDebugInformation || isL1Path) hPassD0Rec.push_back( bookIt("recPassD0_" + myLabel, "Track 2-D impact parameter (0,0,0)(d0) " + myLabel, theD0Parameters));
      hPassD0BeamRec.push_back( bookIt("recPassD0Beam_All", "Track 2-D impact parameter wrt (beam)(d0) ALL", theD0Parameters));
      if (useFullDebugInformation || isL1Path) hPassD0BeamRec.push_back( bookIt("recPassD0Beam_" + myLabel, "Track 2-D impact parameter (beam)(d0) " + myLabel, theD0Parameters));
      
      hPassZ0Rec.push_back( bookIt("recPassZ0_All", "Track Z0 wrt (0,0,0) ALL", theZ0Parameters));
      if (useFullDebugInformation || isL1Path) hPassZ0Rec.push_back( bookIt("recPassZ0_" + myLabel, "Track Z0 (0,0,0) " + myLabel, theZ0Parameters));      
      hPassZ0BeamRec.push_back( bookIt("recPassZ0Beam_All", "Track Z0 wrt (beam) ALL", theZ0Parameters));
      if (useFullDebugInformation || isL1Path) hPassZ0BeamRec.push_back( bookIt("recPassZ0Beam_" + myLabel, "Track Z0 (beam) " + myLabel, theZ0Parameters));

      hPassCharge.push_back( bookIt("recPassCharge_All", "Track Charge  ALL", theChargeParameters));
      if (useFullDebugInformation || isL1Path) hPassCharge.push_back( bookIt("recPassCharge_" + myLabel, "Track Charge  " + myLabel, theChargeParameters));

      hIsolationRec.push_back ( bookIt("recPassIsolation_ALL", "Muon Isolation cone 0.3", theIsolationParameters));
      if (useFullDebugInformation || isL1Path) hIsolationRec.push_back ( bookIt("recPassIsolation_" + myLabel, "Muon Isolation cone 0.3  " + myLabel, theIsolationParameters)); 

        // beamspot filled only once
      hBeamSpotZ0Rec.push_back ( bookIt("recBeamSpotZ0_All", "Z0 of beamspot for this event", theZ0Parameters));

      

      // =======================================================
      // these hisotgrams requite a match, and so will only have
      // L1,L2,L3 histograms and no "all" histogram
      // =======================================================
      
      // hDeltaRMatched.push_back ( bookIt("recDeltaRMatched_All" , "#Delta R between matched HLTCand", theDRParameters));
      if (useFullDebugInformation || isL1Path) hDeltaRMatched.push_back ( bookIt("recDeltaRMatched_" + myLabel, "#Delta R between matched HLTCand", theDRParameters));

      // hChargeFlipMatched.push_back ( bookIt("recChargeFlipMatched_All" , "Charge Flip from hlt to RECO;HLT;Reco", theChargeFlipParameters)); 
      if (useFullDebugInformation || isL1Path) hChargeFlipMatched.push_back ( bookIt("recChargeFlipMatched_" + myLabel, "Charge Flip from hlt to RECO;HLT Charge (-,+);Reco (-,+)", theChargeFlipParameters)); 

      if (useFullDebugInformation || isL1Path) hPassMatchPtRec.push_back( bookIt( "recPassMatchPt_" + myLabel, "Pt of Reco Muon that is matched to Trigger Muon " + myLabel, numBinsInPtHisto, ptBins) );
      //hPtMatchVsPtRec.push_back (bookIt("recPtVsMatchPt" + myLabel, "Reco Pt vs Matched HLT Muon Pt" + myLabel ,  theMaxPtParameters2d) );
      //hEtaMatchVsEtaRec.push_back( bookIt( "recEtaVsMatchEta_" + myLabel, "Reco #eta vs HLT #eta  " + myLabel, theEtaParameters2d) );
      //hPhiMatchVsPhiRec.push_back( bookIt( "recPhiVsMatchPhi_" + myLabel, "Reco #phi vs HLT #phi  " + myLabel, thePhiParameters2d) );
      
      if (useFullDebugInformation || isL1Path) hResoPtAodRec.push_back ( bookIt ("recResoPt_" + myLabel, "TrigSumAOD to RECO P_T resolution", theResParameters));
      if (useFullDebugInformation || isL1Path) hResoEtaAodRec.push_back ( bookIt ("recResoEta_" + myLabel, "TrigSumAOD to RECO #eta resolution", theResParameters));
      if (useFullDebugInformation || isL1Path) hResoPhiAodRec.push_back ( bookIt ("recResoPhi_" + myLabel, "TrigSumAOD to RECO #phi resolution", theResParameters));

      // Cosmic debugging histos
      if (useFullDebugInformation || isL1Path) hMatchedDeltaPhi.push_back ( bookIt( "recDeltaPhiMatched_" + myLabel, "Reco #phi vs HLT #phi  " + myLabel, thePhiParameters0Pi) );
      //hDeltaPhiVsPhi.push_back(bookIt( "recDeltaPhiVsPhi_" + myLabel, "#Delta #phi (reco,hlt) vs HLT #phi  " + myLabel, theDeltaPhiVsPhiParameters) );
      //hDeltaPhiVsZ0.push_back(bookIt( "recDeltaPhiVsZ0_" + myLabel, "#Delta #phi (reco, hlt) vs HLT z0  " + myLabel, theDeltaPhiVsZ0Parameters) );
      //hDeltaPhiVsD0.push_back(bookIt( "recDeltaPhiVsD0_" + myLabel, "#Delta #phi (reco, hlt) vs HLT d0 " + myLabel, theDeltaPhiVsD0Parameters) );
      
      ////////////////////////////////////////////////
      //  RAW Histograms 
      ////////////////////////////////////////////////

      if (useFullDebugInformation) {
        rawMatchHltCandPt.push_back( bookIt( "rawPassPt_All", "Pt of  Reco Muon" ,  numBinsInPtHisto, ptBins) );
        rawMatchHltCandPt.push_back( bookIt( "rawPassPt_" + myLabel, "pt  Reco Muon, if matched to " + myLabel,  numBinsInPtHisto, ptBins) );
      
        rawMatchHltCandEta.push_back( bookIt( "rawPassEta_All", "#eta of Reco Muons", theEtaParameters) );
        rawMatchHltCandEta.push_back( bookIt( "rawPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
      
        rawMatchHltCandPhi.push_back( bookIt( "rawPassPhi_All", "#phi of Reco Muons", thePhiParameters) );
        rawMatchHltCandPhi.push_back( bookIt( "rawPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );
      }
      
      //=================================
      //          2-D Histograms
      //=================================
      
      hPhiVsEtaRec.push_back ( bookIt ("recPhiVsRecEta_All", "Reco #phi vs Reco #eta  ", thePhiEtaParameters2d));
      if (useFullDebugInformation || isL1Path) hPhiVsEtaRec.push_back ( bookIt ("recPhiVsRecEta_" + myLabel, "Reco #phi vs Reco #eta  " +myLabel, thePhiEtaParameters2d));

    }

    // we won't enter this loop if we don't have an hlt label
    // we won't have an hlt label is this is a l1 path
    for ( unsigned int i = 0; i < theHltCollectionLabels.size(); i++ ) {

      if (useOldLabels) {
        myLabel = theHltCollectionLabels[i];
        TString level = ( myLabel.Contains("L2") ) ? "L2" : "L3";
        myLabel = myLabel(myLabel.Index(level),myLabel.Length());
        myLabel = myLabel(0,myLabel.Index("Filtered")+8);
      } else {
        TString tempString = theHltCollectionLabels[i];
        TString level = ( tempString.Contains("L2") ) ? "L2" : "L3";
        myLabel = level + "Filtered";
      }
      
      if ( useMuonFromReco ) {

        // These histos have All, L1, L2, L3
        hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_" + myLabel, "pt of Leading Reco Muon, if matched to " + myLabel, numBinsInPtHisto, ptBins) );     
        hPassEtaRec.push_back( bookIt( "recPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
        hPassPhiRec.push_back( bookIt( "recPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );                

        hPassPtRec.push_back ( bookIt( "recPassPt_" + myLabel, "Pt of  Reco Muon, if matched to " + myLabel, numBinsInPtHisto, ptBins) );
        hPassPtRecExactlyOne.push_back (bookIt( "recPassPtExactlyOne__" + myLabel, "pt of Leading Reco Muon (==1 muon), if matched to " + myLabel, numBinsInPtHisto, ptBins) );

        hPassExaclyOneMuonMaxPtRec.push_back( bookIt("recPassExactlyOneMuonMaxPt_" + myLabel, "pt of Leading Reco Muon in events with exactly one muon match to " + myLabel ,  numBinsInPtHisto, ptBins) );        
        hPhiVsEtaRec.push_back ( bookIt ("recPhiVsRecEta_" + myLabel, "Reco #phi vs Reco #eta  " +myLabel, thePhiEtaParameters2d));

         
        hPassD0Rec.push_back( bookIt("recPassD0_" + myLabel, "Track 2-D impact parameter (Z0) " + myLabel, theD0Parameters));
        hPassD0BeamRec.push_back( bookIt("recPassD0Beam_" + myLabel, "Track 2-D impact parameter (beam)(d0) " + myLabel, theD0Parameters));
        hPassZ0Rec.push_back( bookIt("recPassZ0_" + myLabel, "Track Z0 " + myLabel, theZ0Parameters));
        hPassZ0BeamRec.push_back( bookIt("recPassZ0Beam_" + myLabel, "Track Z0 (0,0,0) " + myLabel, theZ0Parameters));
        hPassCharge.push_back( bookIt("recPassCharge_" + myLabel, "Track Charge  " + myLabel, theChargeParameters));

        hIsolationRec.push_back ( bookIt("recPassIsolation_" + myLabel, "Muon Isolation cone 0.3  " + myLabel, theIsolationParameters)); 
        
        // Match histos only have numHltLabels indices
        hPassMatchPtRec.push_back( bookIt( "recPassMatchPt_" + myLabel, "Pt of Reco Muon that is matched to Trigger Muon " + myLabel, numBinsInPtHisto, ptBins) );

        //hPtMatchVsPtRec.push_back (bookIt("recPtVsMatchPt" + myLabel, "Reco Pt vs Matched HLT Muon Pt" + myLabel ,  theMaxPtParameters2d) );
        //hEtaMatchVsEtaRec.push_back( bookIt( "recEtaVsMatchEta_" + myLabel, "Reco #eta vs HLT #eta  " + myLabel, theEtaParameters2d) );
        //hPhiMatchVsPhiRec.push_back( bookIt( "recPhiVsMatchPhi_" + myLabel, "Reco #phi vs HLT #phi  " + myLabel, thePhiParameters2d) );

        hResoPtAodRec.push_back ( bookIt ("recResoPt_" + myLabel, "TrigSumAOD to RECO P_T resolution", theResParameters));
        hResoEtaAodRec.push_back ( bookIt ("recResoEta_" + myLabel, "TrigSumAOD to RECO #eta resolution", theResParameters));
        hResoPhiAodRec.push_back ( bookIt ("recResoPhi_" + myLabel, "TrigSumAOD to RECO #phi resolution", theResParameters));

        hDeltaRMatched.push_back ( bookIt("recDeltaRMatched_" + myLabel, "#Delta R between matched HLTCand", theDRParameters));
        hChargeFlipMatched.push_back ( bookIt("recChargeFlipMatched_" + myLabel, "Charge Flip from hlt to RECO;HLT (-,+);Reco (-,+)", theChargeFlipParameters)); 

        // cosmic plots

        hMatchedDeltaPhi.push_back ( bookIt( "recDeltaPhiMatched_" + myLabel, "Reco #phi vs HLT #phi  " + myLabel, thePhiParameters0Pi) );  
        //hDeltaPhiVsPhi.push_back(bookIt( "recDeltaPhiVsPhi_" + myLabel, "Reco #phi vs HLT #phi  " + myLabel, theDeltaPhiVsPhiParameters) );
        //hDeltaPhiVsZ0.push_back(bookIt( "recDeltaPhiVsZ0_" + myLabel, "Reco #phi vs HLT #phi  " + myLabel, theDeltaPhiVsZ0Parameters) );
        //hDeltaPhiVsD0.push_back(bookIt( "recDeltaPhiVsD0_" + myLabel, "#Delta #phi (reco, hlt) vs HLT d0 " + myLabel, theDeltaPhiVsD0Parameters) );

        // these candidates are indexed by the number
        // of hlt labels
        allHltCandPt.push_back( bookIt("allHltCandPt_" + myLabel, "Pt of all HLT Muon Cands, for HLT " + myLabel, numBinsInPtHisto, ptBins));     
        allHltCandEta.push_back( bookIt("allHltCandEta_" + myLabel, "Eta of all HLT Muon Cands, for HLT " + myLabel, theEtaParameters));         
        allHltCandPhi.push_back( bookIt("allHltCandPhi_" + myLabel, "Phi of all HLT Muon Cands, for HLT " + myLabel, thePhiParameters));    

        fakeHltCandPt.push_back( bookIt("fakeHltCandPt_" + myLabel, "Pt of fake HLT Muon Cands, for HLT " + myLabel, numBinsInPtHisto, ptBins));     
        fakeHltCandEta.push_back( bookIt("fakeHltCandEta_" + myLabel, "Eta of fake HLT Muon Cands, for HLT " + myLabel, theEtaParameters));         
        fakeHltCandPhi.push_back( bookIt("fakeHltCandPhi_" + myLabel, "Phi of fake HLT Muon Cands, for HLT " + myLabel, thePhiParameters));    
                
        //fakeHltCandEtaPhi.push_back(bookIt("fakeHltCandPhiVsEta_" + myLabel, " AOD #phi vs  #eta for fake HLT Muon Cands, for HLT  " +myLabel, thePhiEtaParameters2d));

        // raw histograms

        if (useFullDebugInformation) {
          rawMatchHltCandPt.push_back( bookIt( "rawPassPt_" + myLabel, "pt  Reco Muon, if matched to " + myLabel,  numBinsInPtHisto, ptBins) );
          rawMatchHltCandEta.push_back( bookIt( "rawPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
          rawMatchHltCandPhi.push_back( bookIt( "rawPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );
        }
      }

    }
  }

}



MonitorElement* HLTMuonMatchAndPlot::bookIt
( TString name, TString title, vector<double> parameters )
{
  LogTrace("HLTMuonVal") << "Directory " << dbe_->pwd() << " Name " << 
                            name << " Title:" << title;
  int nBins  = (int)parameters[0];
  double min = parameters[1];
  double max = parameters[2];

  // this is the 1D hist case
  if (parameters.size() == 3) {
    TH1F *h = new TH1F( name, title, nBins, min, max );
    h->Sumw2();
    MonitorElement * returnedME = dbe_->book1D( name.Data(), h );
    delete h;
    booked1DMonitorElements.push_back(returnedME);
    return returnedME;

    // this is the case for a 2D hist
  } else if (parameters.size() == 6) {

    int nBins2  = (int)parameters[3];
    double min2 = parameters[4];
    double max2 = parameters[5];

    TH2F *h = new TH2F (name, title, nBins, min, max, nBins2, min2, max2);
    h->Sumw2();
    MonitorElement * returnedME = dbe_->book2D (name.Data(), h);
    delete h;    
    return returnedME;

  } else {
    LogInfo ("HLTMuonVal") << "Directory" << dbe_->pwd() << " Name "
                            << name << " had an invalid number of paramters";
    return 0;
  }
  
}

MonitorElement* HLTMuonMatchAndPlot::bookIt
( TString name, TString title, int nbins, float * xBinLowEdges )
{
  LogTrace("HLTMuonVal") << "Directory " << dbe_->pwd() << " Name " << 
                            name << " Title:" << title;

  TH1F *tempHist = new TH1F(name, title, nbins, xBinLowEdges);
  tempHist->Sumw2();
  MonitorElement * returnedME = dbe_->book1D(name.Data(), tempHist);
  delete tempHist;

  booked1DMonitorElements.push_back(returnedME);
  return returnedME;
  
}




int HLTMuonMatchAndPlot::getCharge (int pdgId) {

  int resultCharge =  (pdgId > 0) ? POS_CHARGE : NEG_CHARGE;
  
  return resultCharge;
  
}

TString HLTMuonMatchAndPlot::calcHistoSuffix (string moduleName) {

  TString level = "L3";
  TString myLabel = moduleName; // the return value

  
  if ( myLabel.Contains("L1") ) {
    level = "L1";
  } else if (myLabel.Contains("L2")) {
    level = "L2";
  } else if (myLabel.Contains("L3")) {
    level = "L3";
  }

  if (useOldLabels) {
    myLabel = myLabel(myLabel.Index(level),myLabel.Length());
    myLabel = myLabel(0,myLabel.Index("Filtered")+8);
    
  } else {
    myLabel = level + "Filtered";
  }

  return myLabel;
  
}

void HLTMuonMatchAndPlot::moveOverflow (MonitorElement * myElement) {

  LogTrace ("HLTMuonVal") << "MOVEOVERFLOW" << endl;

  
  // This will handle an arbitrary dimension first/last bin
  // but you should think about how you will interpret this for
  // 2D/3D histos
  // Actually, this can't handle abitrary dimensions.
  int maxBin = myElement->getNbinsX();

  double originalEntries = myElement->getEntries();
  
  LogTrace ("HLTMuonVal") << "==MOVEOVERFLOW==  "
                                << "maxBin = " << maxBin
                                << ", calling underflow"
                                << endl;
  
  myElement->setBinContent(1, myElement->getBinContent(0) + myElement->getBinContent(1));

  LogTrace ("HLTMuonVal") << "reseting underflow to zero" << endl;
  myElement->setBinContent(0,0.0);
  LogTrace ("HLTMuonVal") << "calling overflow" << endl;
  myElement->setBinContent(maxBin,myElement->getBinContent(maxBin) + myElement->getBinContent(maxBin+1));
  LogTrace ("HLTMuonVal") << "seting overflow to zero" << endl;
  myElement->setBinContent(maxBin+1,0.0);

  myElement->setEntries(originalEntries);

}


void HLTMuonMatchAndPlot::getAodTriggerObjectsForModule (edm::InputTag collectionTag,
                                                         edm::Handle<trigger::TriggerEvent> aodTriggerEvent,
                                                         trigger::TriggerObjectCollection trigObjs,
                                                         std::vector<TriggerObject> & foundObjects,
                                                         MuonSelectionStruct muonSelection) {


  //LogTrace ("HLTMuonVal") << "Getting trigger muons for module label = " << collectionTag << endl;
  
  size_t filterIndex   = aodTriggerEvent->filterIndex( collectionTag );
    
  LogTrace ("HLTMuonVal") << "\n\n filterIndex is "
                          << filterIndex;
    
  if ( filterIndex < aodTriggerEvent->sizeFilters() ) {
    const Keys &keys = aodTriggerEvent->filterKeys( filterIndex );

    LogTrace ("HLTMuonVal") << "\n\nGot keys";
    LogTrace ("HLTMuonVal") << "Key size is " << keys.size();
                              
    // The keys are apparently pointers into the trigger
    // trigObjs collections
    // Use the key's to look up the particles for the
    // filter module that you're using 
      
    for ( size_t j = 0; j < keys.size(); j++ ){
      TriggerObject foundObject = trigObjs[keys[j]];

      // This is the trigger object. Apply your filter to it!
      LogTrace ("HLTMuonVal") << "Testing to see if object in key passes selection"
                              << endl ;
        
      if (muonSelection.hltMuonSelector(foundObject)){
        
        LogTrace ("HLTMuonVal") << "OBJECT FOUND!!! - Storing a trigger object with id = "              
                                << foundObject.id() 
                                << ", eta = " << foundObject.eta()
                                << ", pt = " << foundObject.pt()
                                << ", custom name = " << muonSelection.customLabel
                                << "\n\n" << endl;
        //l1Particles.push_back( trigObjs[keys[j]].particle().p4() );
        foundObjects.push_back( foundObject );
      }
    }
  }

  
  
}
