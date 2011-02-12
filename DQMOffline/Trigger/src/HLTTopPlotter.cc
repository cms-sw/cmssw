 /** \file DQMOffline/Trigger/HLTTopPlotter.cc
 *
 *  Muon HLT Offline DQM plotting code
 *  This object will make occupancy/efficiency plots for a
 *  specific set of conditions:
 *    1. A set of selection cuts
 *    2. A trigger name
 *  
 *  $Author: slaunwhj $
 *  $Date: 2009/11/13 12:39:32 $
 *  $Revision: 1.2 $
 */



#include "DQMOffline/Trigger/interface/HLTTopPlotter.h"




#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"


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

//using HLTMuonMatchAndPlot::MatchStruct;

typedef std::vector< edm::ParameterSet > Parameters;
typedef std::vector<reco::Muon> MuonCollection;

// const int numCones     = 3;
// const int numMinPtCuts = 1;
// double coneSizes[] = { 0.20, 0.24, 0.30 };
// double minPtCuts[] = { 0. };


/// Constructor
HLTTopPlotter::HLTTopPlotter
( const ParameterSet& pset, string triggerName, vector<string> moduleNames,
  MuonSelectionStruct inputSelection, string customName,
  vector<string> validTriggers,
  const edm::Run & currentRun,
  const edm::EventSetup & currentEventSetup)
  : HLTMuonMatchAndPlot(pset, triggerName, moduleNames, inputSelection, customName, validTriggers, currentRun, currentEventSetup)
    
{

  CaloJetInputTag       = pset.getParameter<edm::InputTag>("CaloJetInputTag");
  EtaCut_               = pset.getUntrackedParameter<double>("EtaCut");
  PtCut_                = pset.getUntrackedParameter<double>("PtCut");   
  NJets_                = pset.getUntrackedParameter<int>("NJets");   
  theJetMParameters    = pset.getUntrackedParameter< vector<double> >("JetMParameters");

  
  LogTrace ("HLTMuonVal") << "\n\n Inside HLTTopPlotter Constructor";
  LogTrace ("HLTMuonVal") << "The trigger name is " << triggerName
                          << "and we've done all the other intitializations";

  LogTrace ("HLTMuonVal") << "exiting constructor\n\n";

}



void HLTTopPlotter::finish()
{

  // you could do something else in here
  // but for now, just do what the base class
  // would have done
  
  HLTMuonMatchAndPlot::finish();
}



void HLTTopPlotter::analyze( const Event & iEvent )
{

  LogTrace ("HLTMuonVal") << "Inside of TopPlotter analyze method!"
                          << "calling my match and plot module's analyze..."
                          << endl;

  // Make sure you are valid before proceeding


  // Do some top specific selection, then call the muon matching
  // if the event looks top-like

  LogTrace ("HLTMuonVal") << "Do top-specific selection" << endl;
  

  // get calo jet collection
  Handle<CaloJetCollection> jetsHandle;
  iEvent.getByLabel(CaloJetInputTag, jetsHandle);

  int n_jets_20 = 0;
  CaloJetCollection selectedJets;
  
  if (jetsHandle.isValid()) {
    LogTrace ("HLTMuonVal") << "Looking in jet collection" << endl;
    //Jet Collection to use
    
    // Raw jets
    const CaloJetCollection *jets = jetsHandle.product();
    CaloJetCollection::const_iterator jet;
    
    //int n_jets_20=0;

    // The parameters for the n jets should be 

    
    for (jet = jets->begin(); jet != jets->end(); jet++){        
      // if (fabs(jet->eta()) <2.4 && jet->et() > 20) n_jets_20++; 
      if (fabs(jet->eta()) <EtaCut_ && jet->et() > PtCut_) {
        n_jets_20++;
        selectedJets.push_back((*jet));
      }
      
    } 
    
  }
  
 

  // sort the result
  sortJets(selectedJets);

  
  
  LogTrace ("HLTMuonVal") << "Number of jets in this event = "
                          << n_jets_20
                          << endl;

 // if (n_jets_20 <= 1 ) {
  if (n_jets_20 < NJets_ ) {
    LogTrace ("HLTMuonVal") << "Not enought jets in this event, skipping it"
                            << endl;

    return;
  }
  
 

  /////////////////////////////////////////////////
  //
  //     Call the other analyze method
  //
  /////////////////////////////////////////////////
  


  LogTrace("HLTMuonVal") << "Calling analyze for muon ana" << endl;
  HLTMuonMatchAndPlot::analyze(iEvent);


  LogTrace ("HLTMuonVal") << "TOPPLOT: returned from muon ana, now in top module"
                          << endl
                          << "muon ana stored size rec muons = "
    //<< myMuonAna->recMatches.size()
                          << endl;
  
  vector<HLTMuonMatchAndPlot::MatchStruct>::const_iterator iRecMuon;

  int numCands = 0;
  for ( unsigned int i  = 0;
        i < recMatches.size();
        i++ ) {


    LogTrace ("HLTMuonVal") << "Cand " << numCands
                            << ", Pt = "
                            << recMatches[i].recCand->pt()
                            << ", eta = "
                            << recMatches[i].recCand->eta()
                            << ", phi = "
                            << recMatches[i].recCand->phi()
                            << endl;

    
    double deltaRLeadJetLep = reco::deltaR (recMatches[i].recCand->eta(), recMatches[i].recCand->phi(),
                                       selectedJets[0].eta(), selectedJets[0].phi());

    ////////////////////////////////////////////
    //
    //   Fill Plots for All
    //
    ////////////////////////////////////////////
    
    hDeltaRMaxJetLep[0]->Fill(deltaRLeadJetLep);
    hJetMultip[0]->Fill(n_jets_20);


    ////////////////////////////////////////////
    //
    //   Fill Plots for L1
    //
    ////////////////////////////////////////////

    
    if ( (recMatches[i].l1Cand.pt() > 0) && ((useFullDebugInformation) || (isL1Path)) ) {
      hDeltaRMaxJetLep[1]->Fill(deltaRLeadJetLep);
      hJetMultip[1]->Fill(n_jets_20);
    }

    ////////////////////////////////////////////
    //
    //   Fill Plots for HLT
    //
    ////////////////////////////////////////////
    
    for ( size_t j = 0; j < recMatches[i].hltCands.size(); j++ ) {
      if ( recMatches[i].hltCands[j].pt() > 0 ) {
        // you've found it!
        hDeltaRMaxJetLep[j+HLT_PLOT_OFFSET]->Fill(deltaRLeadJetLep);
	hJetMultip[j+HLT_PLOT_OFFSET]->Fill(n_jets_20);
        
      }
    }
    

    
    numCands++;
  }
  

  LogTrace ("HLTMuonVal") << "-----End of top plotter analyze method-----" << endl;
} // Done filling histograms




void HLTTopPlotter::begin() 
{

  TString myLabel, newFolder;
  vector<TH1F*> h;

  LogTrace ("HLTMuonVal") << "Inside begin for top analyzer" << endl;

  
  LogTrace ("HLTMuonVal") << "Calling begin for muon analyzer" << endl;
  HLTMuonMatchAndPlot::begin();

  LogTrace ("HLTMuonVal") << "Continuing with top analyzer begin" << endl;

  if ( dbe_ ) {
    dbe_->cd();
    dbe_->setCurrentFolder("HLT/Muon");

    
    // JMS I think this is trimming all L1 names to
    // to be L1Filtered
    myLabel = theL1CollectionLabel;
    myLabel = myLabel(myLabel.Index("L1"),myLabel.Length());
    myLabel = myLabel(0,myLabel.Index("Filtered")+8);


    // JMS Old way of doing things
    //newFolder = "HLT/Muon/Distributions/" + theTriggerName;
    newFolder = "HLT/Muon/Distributions/" + theTriggerName + "/" + mySelection.customLabel;

    
    
    dbe_->setCurrentFolder( newFolder.Data() );
    
    vector<string> binLabels;
    binLabels.push_back( theL1CollectionLabel.c_str() );
    for ( size_t i = 0; i < theHltCollectionLabels.size(); i++ )
      binLabels.push_back( theHltCollectionLabels[i].c_str() );


    //------- Define labels for plots -------
    
    if (useOldLabels) { 
      myLabel = theL1CollectionLabel;
      myLabel = myLabel(myLabel.Index("L1"),myLabel.Length());
      myLabel = myLabel(0,myLabel.Index("Filtered")+8);
    } else {
      myLabel = "L1Filtered";
    }

    //------ Definte the plots themselves------------------------


    //////////////////////////////////////////////////////////////
    //
    //         ALL + L1 plots 
    //
    //////////////////////////////////////////////////////////////
    
    hDeltaRMaxJetLep.push_back (bookIt("topDeltaRMaxJetLep_All", "delta R between muon and highest pt jet", theDRParameters));
    if (useFullDebugInformation || isL1Path) hDeltaRMaxJetLep.push_back (bookIt("topDeltaRMaxJetLep_" + myLabel, "delta R between muon and highest pt jet", theDRParameters));

   hJetMultip.push_back (bookIt("topJetMultip_All", "Jet multiplicity", theJetMParameters));
    if (useFullDebugInformation || isL1Path) hJetMultip.push_back(bookIt("topJetMultip_" + myLabel, "Jet multiplicity", theJetMParameters));


    //////////////////////////////////////////////////////////////
    //
    //         ALL + L1 plots 
    //
    //////////////////////////////////////////////////////////////
    
    
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
      } // end if useOldLabels
    

      // Book for L2, L3
      hDeltaRMaxJetLep.push_back (bookIt("topDeltaRMaxJetLep_" + myLabel, "delta R between muon and highest pt jet", theDRParameters)) ;
      hJetMultip.push_back (bookIt("topJetMultip_" + myLabel, "Jet Multiplicity",   theJetMParameters)) ;
      
      
      
    }// end for each collection label

  }// end if dbe_ exists

}// end begin method



///////////////////////////////////////////////////////
//
//  Extra methods
//
///////////////////////////////////////////////////////



// ---------------   Sort a collection of Jets -----------


void HLTTopPlotter::sortJets (CaloJetCollection & theJets) {


  LogTrace ("HLTMuonVal") << "Sorting Jets" << endl;
  
  // bubble sort jets
  
  for ( unsigned int iJet = 0;
        iJet < theJets.size();          
        iJet ++) {

    for ( unsigned int jJet = iJet;
          jJet < theJets.size();
          jJet ++ ){

      if ( theJets[jJet].et() > theJets[iJet].et() ) {
        reco::CaloJet tmpJet = theJets[iJet];
        theJets[iJet] = theJets[jJet];
        theJets[jJet] = tmpJet;        
      }
      
    }// end for each jJet

  }// end for each iJet

  for ( unsigned int iJet = 0;
        iJet != theJets.size();
        iJet ++ ) {

    LogTrace ("HLTMuonVal") << "Jet # " << iJet
                            << "  Et =  " << theJets[iJet].et()
                            << endl;

  }
  
}


