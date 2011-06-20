 /** \file DQMOffline/Trigger/HLTMuonBPAG.cc
 *
 *  Muon HLT Offline DQM plotting code
 *  This object will make occupancy/efficiency plots for a
 *  specific set of conditions:
 *    1. A set of selection cuts
 *    2. A trigger name
 *  
 *  $Author: slaunwhj $
 *  $Date: 2010/02/22 16:16:46 $
 *  $Revision: 1.7 $
 */



#include "DQMOffline/Trigger/interface/HLTMuonBPAG.h"


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
HLTMuonBPAG::HLTMuonBPAG
( const ParameterSet& pset, string triggerName, vector<string> moduleNames,
  MuonSelectionStruct probeSelection,
  MuonSelectionStruct inputTagSelection, string customName,
  vector<string> validTriggers,
  const edm::Run & currentRun,
  const edm::EventSetup & currentEventSetup)  
  : HLTMuonMatchAndPlot(pset, triggerName, moduleNames, probeSelection, customName, validTriggers, currentRun, currentEventSetup),
    tagSelection(inputTagSelection)
    
{


  LogTrace ("HLTMuonVal") << "\n\n Inside HLTMuonBPAG Constructor";
  LogTrace ("HLTMuonVal") << "The trigger name is " << triggerName
                          << "and we've done all the other intitializations";

  LogTrace ("HLTMuonVal") << "exiting constructor\n\n";

  ALLKEY = "ALL";

  // pick up the mass parameters
  theMassParameters = pset.getUntrackedParameter< vector<double> >("MassParameters");

}



void HLTMuonBPAG::finish()
{

  // you could do something else in here
  // but for now, just do what the base class
  // would have done
  
  HLTMuonMatchAndPlot::finish();
}



void HLTMuonBPAG::analyze( const Event & iEvent )
{

  LogTrace ("HLTMuonVal") << "Inside of BPAG analyze method!"
                          << "calling my match and plot module's analyze..."
                          << endl;

  // Make sure you are valid before proceeding


  // Do some top specific selection, then call the muon matching
  // if the event looks top-like

  
  /////////////////////////////////////////////////
  //
  //     Call the other analyze method
  //
  /////////////////////////////////////////////////
  


  //LogTrace("HLTMuonVal") << "Calling muon selection for muon ana" << endl;

  // Call analyze to get everything

  //HLTMuonMatchAndPlot::analyze(iEvent);

  // separate calls 

  LogTrace ("HLTMuonVal") << "BPAG: calling subclass matching routine" << endl;
  
  // select and match muons
  selectAndMatchMuons(iEvent, recMatches, hltFakeCands);

  
  LogTrace ("HLTMuonVal") << "BPAG: returned from muon ana, now in BAPG module"
                          << endl
                          << "  muon ana stored size probe muons = "
                          << recMatches.size() 
                          << "  tag muons size = "
                          << tagRecMatches.size()
                          << endl;
  
  //  vector<HLTMuonMatchAndPlot::MatchStruct>::const_iterator iRecMuon;

  //int numCands = 0;

  for (unsigned iTag = 0;
       iTag < tagRecMatches.size();
       iTag ++) {

    // We should check to see that
    // each tag passed a tag trigger

    bool passedHLT = false;

    LogTrace ("HLTMuonVal") << "CRASH: tagRecMatches[iTag].hltCands.size() =  "
                            << theHltCollectionLabels.size() << endl
                            << "CRASH: theHltCollectionLabels.size() =  "
                            << theHltCollectionLabels.size()
                            << endl;

//     cout <<  "==========================================================================" << endl
//          <<  "  Run =  " << iEvent.id().run() << "  Event =  " << iEvent.id().event() << endl
//          <<  "  tagRecMatches[iTag].hltCands.size() = " << tagRecMatches[iTag].hltCands.size() << endl
//          <<  "  theHltCollectionLabels.size() = " << theHltCollectionLabels.size() << endl
//          <<  ""  << endl;
      

      
    if ( theHltCollectionLabels.size() <= tagRecMatches[iTag].hltCands.size()) {
      for ( size_t jCollLabel = 0; jCollLabel < theHltCollectionLabels.size(); jCollLabel++ ) {
        if ( tagRecMatches[iTag].hltCands[jCollLabel].pt() > 0 ) {        
          passedHLT = true;
        }
      }
    }

    LogTrace ("HLTMuonVal") << "===BPAG=== Did Tag # " << iTag << " pass the trigger? "
                            << ((passedHLT) ? "YES" : "NO") << endl
                            << "    if no, then we will skip it as a tag..."  << endl;
    
    if (!passedHLT) continue;
    
    for ( unsigned int iProbe  = 0;
          iProbe < recMatches.size();
          iProbe++ ) {
      
      
      LogTrace ("HLTMuonVal") << "Probe = " << iProbe << endl
                              << " Pt = " << endl
                              << recMatches[iProbe].recCand->pt()
                              << " eta = " << endl
                              << recMatches[iProbe].recCand->eta()
                              << " phi = " << endl
                              << recMatches[iProbe].recCand->phi()
                              << endl << endl
                              << "Tag = " << iTag
                              << " Pt = " << endl
                              << tagRecMatches[iTag].recCand->pt()
                              << " eta = " << endl
                              << tagRecMatches[iTag].recCand->eta()
                              << " phi = " << endl
                              << tagRecMatches[iTag].recCand->phi()
                              << endl;

      if ( recMatches[iProbe].recCand->charge() * tagRecMatches[iTag].recCand->charge() > 0 ){
        LogTrace ("HLTMuonVal") << "Tag and Probe don't have opp charges, skipping to next probe" 
                                << endl;

        continue;
      }

      LorentzVector tagPlusProbe = (recMatches[iProbe].recCand->p4() + tagRecMatches[iTag].recCand->p4());

      double invariantMass = tagPlusProbe.mass();
      double ptProbe       = recMatches[iProbe].recCand->pt();
      double etaProbe      = recMatches[iProbe].recCand->eta();
      double phiProbe      = recMatches[iProbe].recCand->phi();
      
      ////////////////////////////////////////////
      //
      //   Fill Plots for All
      //
      ////////////////////////////////////////////

      diMuonMassVsPt[ALLKEY]->Fill(ptProbe, invariantMass);
      diMuonMassVsEta[ALLKEY]->Fill(etaProbe, invariantMass);
      diMuonMassVsPhi[ALLKEY]->Fill(phiProbe, invariantMass);
      
      diMuonMass[ALLKEY]->Fill(invariantMass);

      ////////////////////////////////////////////
      //
      //   Fill Plots for L1
      //
      ////////////////////////////////////////////

    
      if ( (recMatches[iProbe].l1Cand.pt() > 0) && ((useFullDebugInformation) || (isL1Path)) ) {
        TString myLabel = calcHistoSuffix(theL1CollectionLabel);
        
        diMuonMassVsPt[myLabel]->Fill(ptProbe, invariantMass);
        diMuonMassVsEta[myLabel]->Fill(etaProbe, invariantMass);
        diMuonMassVsPhi[myLabel]->Fill(phiProbe, invariantMass);
        
        diMuonMass[myLabel]->Fill(invariantMass);
      }
      
      ////////////////////////////////////////////
      //
      //   Fill Plots for HLT
      //
      ////////////////////////////////////////////

      LogTrace ("HLTMuonVal") << "The size of the HLT collection labels =   " << theHltCollectionLabels.size()
                              << ", and the size of recMatches[" << iProbe << "].hltCands = "
                              << recMatches[iProbe].hltCands.size()
                              << endl;
        
      for ( size_t j = 0; j < theHltCollectionLabels.size(); j++ ) {
        if ( recMatches[iProbe].hltCands[j].pt() > 0 ) {
          TString myLabel = calcHistoSuffix (theHltCollectionLabels[j]);

          LogTrace ("HLTMuonVal") << "filling plot ... Looking up the plot label " << myLabel
                                  << endl;

          // Do the existence check for each plot
          // print out the results only for the first one
          
          if (diMuonMassVsPt.find(myLabel) != diMuonMassVsPt.end()){
            LogTrace ("HLTMuonVal") << "Found a plot corresponding to label = "
                                    << myLabel << endl;

            diMuonMassVsPt[myLabel]->Fill(ptProbe, invariantMass);
            
            
          } else {
            LogTrace ("HLTMuonVal") << "Didn't find a plot corresponding to your label" << endl;
          }

          if (diMuonMass.find(myLabel) != diMuonMass.end()) 
            diMuonMass[myLabel]->Fill(invariantMass);

          if (diMuonMassVsEta.find(myLabel) != diMuonMassVsEta.end())
            diMuonMassVsEta[myLabel]->Fill(etaProbe, invariantMass);

          if (diMuonMassVsPhi.find(myLabel) != diMuonMassVsPhi.end())
            diMuonMassVsPhi[myLabel]->Fill(phiProbe, invariantMass);
          
        }
      }
    

    
      //numCands++;
    } // end loop over probes
  } // end loop over tags

  LogTrace ("HLTMuonVal") << "-----End of BPAG plotter analyze method-----" << endl;
} // Done filling histograms




void HLTMuonBPAG::begin() 
{

  TString myLabel, newFolder;
  vector<TH1F*> h;

  LogTrace ("HLTMuonVal") << "Inside begin for BPAG analyzer" << endl;

  
  //LogTrace ("HLTMuonVal") << "Calling begin for muon analyzer" << endl;
  //HLTMuonMatchAndPlot::begin();

  //LogTrace ("HLTMuonVal") << "Continuing with top analyzer begin" << endl;

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

    //----- define temporary vectors that
    //----- give you dimensions

    
    // vector<double>  massVsPtBins; // not used 
    //     massVsPtBins.push_back(5); // pt from 0 to 20 in 5 bins
    //     massVsPtBins.push_back(0.0);
    //     massVsPtBins.push_back(20.0);
    //     massVsPtBins.push_back(50); // mass: 10 bins from 0 to 6
    //     massVsPtBins.push_back(0.0);
    //     massVsPtBins.push_back(6.0);

    int nPtBins = ((int) thePtParameters.size()) - 1;
    int nMassBins = (theMassParameters.size() > 0) ? ((int)theMassParameters[0]) : 50;
    double minMass = (theMassParameters.size() > 1) ? theMassParameters[1] : 0;
    double maxMass = (theMassParameters.size() > 2) ? theMassParameters[2] : 6;;

    double ptBinLowEdges[100]; // set to a large maximum

    unsigned maxPtBin = (thePtParameters.size() > 100) ? thePtParameters.size() : 100;
    
    for (unsigned i = 0; i < maxPtBin; i++)
      ptBinLowEdges[i] = thePtParameters[i];
    
    vector<double> evenPtBins;
    evenPtBins.push_back(10);
    evenPtBins.push_back(0);
    evenPtBins.push_back(20);

    
    



    vector<double> massVsEtaBins;
    massVsEtaBins.push_back(theEtaParameters[0]); // |eta| < 2.1 in 5 bins
    massVsEtaBins.push_back(theEtaParameters[1]);
    massVsEtaBins.push_back(theEtaParameters[2]);
    massVsEtaBins.push_back(nMassBins);
    massVsEtaBins.push_back(minMass); // mass: 10 bins from 0 to 6
    massVsEtaBins.push_back(maxMass);

    vector<double> massVsPhiBins;
    massVsPhiBins.push_back(thePhiParameters[0]); // -pi < phi < pi  in 5 bins
    massVsPhiBins.push_back(thePhiParameters[1]);
    massVsPhiBins.push_back(thePhiParameters[2]);
    massVsPhiBins.push_back(nMassBins);
    massVsPhiBins.push_back(minMass); // mass: 10 bins from 0 to 6
    massVsPhiBins.push_back(maxMass);

    

    vector<double> massBins;
    massBins.push_back(nMassBins);
    massBins.push_back(minMass);
    massBins.push_back(maxMass);

    //////////////////////////////////////////////////////////////
    //
    //         ALL + L1 plots 
    //
    //////////////////////////////////////////////////////////////

    
    //diMuonMassVsPt[ALLKEY] = bookIt("diMuonMassVsPt_All", "Mass Vs Probe Pt", massVsPtBins);
    diMuonMassVsPt[ALLKEY] =  book2DVarBins("diMuonMassVsPt_All", "Mass Vs Probe Pt; Pt; Mass", nPtBins, ptBinLowEdges, nMassBins, minMass, maxMass);
    diMuonMassVsEta[ALLKEY] = bookIt("diMuonMassVsEta_All", "Mass Vs Probe Eta; #eta ; Mass", massVsEtaBins);    
    diMuonMassVsPhi[ALLKEY] = bookIt("diMuonMassVsPhi_All", "Mass Vs Probe Phi", massVsPhiBins);
    
    diMuonMass[ALLKEY] = bookIt("diMuonMass_All", "Mass of Dimuons; Mass", massBins);

    probeMuonPt[ALLKEY] = bookIt("probeMuonPt_All", "Probe Muon PT; Probe Pt", evenPtBins);
    
    
    if (useFullDebugInformation || isL1Path) {
      diMuonMassVsPt[myLabel] = book2DVarBins("diMuonMassVsPt_" + myLabel, "Mass Vs Probe Pt; Pt; Mass", nPtBins, ptBinLowEdges, nMassBins, minMass, maxMass);
      diMuonMassVsEta[myLabel] = bookIt("diMuonMassVsEta_" + myLabel, "Mass Vs Probe Eta; #eta; Mass " + myLabel, massVsEtaBins);
      diMuonMassVsPhi[myLabel] = bookIt("diMuonMassVsPhi_" + myLabel, "Mass Vs Probe Phi; #phi; Mass " + myLabel, massVsPhiBins);
 
      diMuonMass[myLabel] = bookIt("diMuonMass_" + myLabel, "Mass of Dimuons; mass  " + myLabel, massBins);
      probeMuonPt[myLabel] = bookIt("probeMuonPt_" + myLabel, "Probe Muon PT; Pt" + myLabel, evenPtBins);
      
    }


    //////////////////////////////////////////////////////////////
    //
    //         HLT  plots 
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
      diMuonMassVsPt[myLabel] = book2DVarBins("diMuonMassVsPt_" + myLabel, "Mass Vs Probe Pt; Pt; Mass" + myLabel, nPtBins, ptBinLowEdges, nMassBins, minMass, maxMass);
      diMuonMassVsEta[myLabel] = bookIt("diMuonMassVsEta_" + myLabel, "Mass Vs Probe Eta; #eta; Mass " + myLabel, massVsEtaBins);
      diMuonMassVsPhi[myLabel] = bookIt("diMuonMassVsPhi_" + myLabel, "Mass Vs Probe Phi; #phi; Mass " + myLabel, massVsPhiBins);

      diMuonMass[myLabel] = bookIt("diMuonMass_" + myLabel, "Mass of Dimuons; Mass  " + myLabel, massBins);
      probeMuonPt[myLabel] = bookIt("probeMuonPt_" + myLabel, "Probe Muon PT; Pt "+ myLabel, evenPtBins);
      
    }// end for each collection label

    map<TString, MonitorElement*>::const_iterator iPlot;

    LogTrace ("HLTMuonVal") << "BPAG::begin dumping some plot names " << endl;
    for (iPlot = diMuonMassVsPt.begin();
         iPlot != diMuonMassVsPt.end();
         iPlot++){

      LogTrace("HLTMuonVal") << "BPAG:     PLOT Key = " << iPlot->first << endl;
    }

    
  }// end if dbe_ exists

}// end begin method

//////////////////////////////////////////////////////////
//
// Redefine what it means to do match and select
//
///////////////////////////////////////////////////////////

bool HLTMuonBPAG::selectAndMatchMuons(const edm::Event & iEvent,
                                      std::vector<MatchStruct> & argRecMatches,
                                      std::vector< std::vector<HltFakeStruct> > & argHltFakeCands
                                      ) {

  // Initialize this match
  argRecMatches.clear();
  tagRecMatches.clear();
  
  // call select with a reco muon PROBE selection
  HLTMuonMatchAndPlot::selectAndMatchMuons(iEvent, argRecMatches, argHltFakeCands, mySelection);

  

  // call select with a reco muon tag selection argument
  // First, intialize a probe selection
  // Some day this should come from the driver
  
  //StringCutObjectSelector<Muon> tempRecoSelector("pt > 1 && abs(eta) < 1.4");
  //StringCutObjectSelector<TriggerObject> tempHltSelector("pt > 1 && abs(eta) < 1.4");
  string customName = "bpagTag";
  //double d0Cut = 2.0;
  //double z0Cut = 50;
  string trkCol = "innerTrack";
  std::vector<std::string> reqTrigs;

  //MuonSelectionStruct tagSelection(tempRecoSelector, tempHltSelector,
  //                                 customName, d0Cut, z0Cut, trkCol, reqTrigs);
  
  //==========================
  //  tagRecMatches
  //  and tagHltFakeCands
  //  are private members 
  //==========================
  
  HLTMuonMatchAndPlot::selectAndMatchMuons(iEvent, tagRecMatches, tagHltFakeCands, tagSelection);


  // now you have two vectors, one with probes and one with tags
  

  LogTrace ("HLTMuonVal") << "Printing tags and probes!!!"
                          << "NTAGS   =   " << tagRecMatches.size() << endl
                          << "NPROBES =   " << argRecMatches.size() << endl
                          << endl;

  
  
  for (unsigned iProbe = 0;
       iProbe < argRecMatches.size();
       iProbe++) {

    LogTrace ("HLTMuonVal") << "Probe # " << iProbe 
                            << "  PT = " << argRecMatches[iProbe].recCand->pt()
                            << "  ETA = " << argRecMatches[iProbe].recCand->eta()
                            << "  PHI = " << argRecMatches[iProbe].recCand->phi()
                            << endl;
    

  }

  for (unsigned iTag = 0;
       iTag < tagRecMatches.size();
       iTag++) {

    LogTrace ("HLTMuonVal") << "Tag # " << iTag 
             << "  PT = " << tagRecMatches[iTag].recCand->pt()
             << "  ETA = " << tagRecMatches[iTag].recCand->eta()
             << "  PHI = " << tagRecMatches[iTag].recCand->phi()
             << endl;
    

  }

  // you may have overlapping tags & probes
  // but you've sucessfully searched for them, so return true

  return true;
  
}


///////////////////////////////////////////////////////
//
//  Extra methods
//
///////////////////////////////////////////////////////



MonitorElement* HLTMuonBPAG::book2DVarBins (TString name, TString title, int nBinsX, double * xBinLowEdges, int nBinsY, double yMin, double yMax) {

  TH2F *tempHist = new TH2F(name, title, nBinsX, xBinLowEdges, nBinsY, yMin, yMax);
  tempHist->Sumw2();
  MonitorElement * returnedME = dbe_->book2D(name.Data(), tempHist);
  delete tempHist;
  return returnedME;

}
