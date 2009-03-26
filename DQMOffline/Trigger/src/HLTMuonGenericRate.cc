 /** \file HLTMuonGenericRate.cc
 *  Get L1/HLT efficiency/rate plots
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2009/02/27 13:11:32 $
 *  $Revision: 1.1 $
 */


#include "DQMOffline/Trigger/interface/HLTMuonGenericRate.h"
#include "DQMOffline/Trigger/interface/AnglesUtil.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

// For storing calorimeter isolation info in the ntuple
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "TPRegexp.h"
#include <iostream>

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;

typedef std::vector< edm::ParameterSet > Parameters;

const int numCones     = 3;
const int numMinPtCuts = 1;
double coneSizes[] = { 0.20, 0.24, 0.30 };
double minPtCuts[] = { 0. };


/// Constructor
HLTMuonGenericRate::HLTMuonGenericRate
( const ParameterSet& pset, string triggerName, vector<string> moduleNames, string myRecoCollection )
{


  LogTrace ("HLTMuonVal") << "\n\n Inside HLTMuonGenericRate Constructor";
  LogTrace ("HLTMuonVal") << "The trigger name is " << triggerName
                          << " and the module names are listed";

  for (vector<string>::iterator iMod = moduleNames.begin();
       iMod != moduleNames.end(); iMod++){
    LogTrace ("HLTMuonVal") << (*iMod);
  }
    
  
  theHltProcessName  = pset.getParameter<string>("HltProcessName");
  theNumberOfObjects = ( TString(triggerName).Contains("Double") ) ? 2 : 1;
  theTriggerName     = triggerName;

  LogTrace ("HLTMuonVal") << "\n\n Getting AOD switch + lables \n\n";
  
  useAod         = pset.getUntrackedParameter<bool>("UseAod");
  theAodL1Label  = pset.getUntrackedParameter<string>("AodL1Label");
  theAodL2Label  = pset.getUntrackedParameter<string>("AodL2Label");

  // JMS Added a method to make standalone histogram output
  createStandAloneHistos = pset.getUntrackedParameter<bool>("createStandAloneHistos");
  histoFileName = pset.getUntrackedParameter<string> ("histoFileName");

  theHltCollectionLabels.clear();
  TPRegexp l1Regexp("L1.*Filtered");
  for ( size_t i = 0; i < moduleNames.size(); i++ ) {
    string module = moduleNames[i];

    LogTrace ("HLTMuonVal") << "Considering Module named "
                            << module;
    
    if ( TString(module).Contains(l1Regexp) ) {
      theL1CollectionLabel = module;
      LogTrace ("HLTMuonVal") << "... module is L1 collection";      
    } else if ( TString(module).Contains("Filtered") ) {
      theHltCollectionLabels.push_back(module);
      LogTrace ("HLTMuonVal") << "... module is HLT collection";
    }
  }

  LogTrace ("HLTMuonVal") << "Skipping special AOD handling";
  // if ( useAod ) {
//     LogTrace ("HLTMuonVal") << "Storing AOD labels";

//     LogTrace ("HLTMuonVal") << "The AodL1Label is "
//                             << theAodL1Label
//                             << "\nThe size of theHltCollectionLabels is "
//                             << theHltCollectionLabels.size();
    
//     theL1CollectionLabel = theAodL1Label;

//     // If you have a hltCollection, then do this tweak
//     // If not, then leave it alone
//     if ( theHltCollectionLabels.size() != 0){
//       string & finalLabel    = theHltCollectionLabels.back();
//       LogTrace ("HLTMuonVal") << "Final Label is " << finalLabel;
    
//       theHltCollectionLabels.clear();
//       theHltCollectionLabels.push_back( theAodL2Label );
//       theHltCollectionLabels.push_back( finalLabel );
//     }

//     LogTrace ("HLTMuonVal") << "Done storing labels\n\n";
//   }

  numHltLabels   = theHltCollectionLabels.size();
  isIsolatedPath = ( numHltLabels == 4 ) ? true : false;

  theGenLabel          = pset.getUntrackedParameter<string>("GenLabel" ,"");
  // old way
  // theRecoLabel         = pset.getUntrackedParameter<string>("RecoLabel","");

  // New way
  LogTrace ("HLTMuonVal") << "\n\nThe RECO collection for this GenericRate is "
                          << myRecoCollection;
  
  theRecoLabel = myRecoCollection;
  
  useMuonFromGenerator = ( theGenLabel  == "" ) ? false : true;
  useMuonFromReco      = ( theRecoLabel == "" ) ? false : true;

  theMaxPtParameters = pset.getParameter< vector<double> >("MaxPtParameters");
  thePtParameters    = pset.getParameter< vector<double> >("PtParameters");
  theEtaParameters   = pset.getParameter< vector<double> >("EtaParameters");
  thePhiParameters   = pset.getParameter< vector<double> >("PhiParameters");

  theResParameters = pset.getParameter < vector<double> >("ResParameters");

  //highPtTrackCollection = pset.getParameter <string> ("highPtTrackCollection");
  
  // Duplicate the pt parameters for some 2D histos
  for(int i =0; i < 2; i++){
    for (std::vector<double>::const_iterator iNum = theMaxPtParameters.begin();
         iNum != theMaxPtParameters.end();
         iNum++){
      
      // if this is the # of bins, then
      // double the number of bins.
      if (iNum == theMaxPtParameters.begin()){
        theMaxPtParameters2d.push_back(2*(*iNum));
      } else {
        theMaxPtParameters2d.push_back((*iNum));
      }
    }
  }

  // Duplicate the pt parameters for some 2D histos
  for(int i =0; i < 2; i++){
    for (std::vector<double>::const_iterator iNum = theEtaParameters.begin();
         iNum != theEtaParameters.end();
         iNum++){
      // if this is the nBins param, double it
      if (iNum ==  theEtaParameters.begin()){
        theEtaParameters2d.push_back(3*(*iNum));      
      } else {
        theEtaParameters2d.push_back(*iNum);                   
      }
      
      // also fill the eta/phi plot parameters
      // but don't worry about doubleing bins
      if (i < 1){
        thePhiEtaParameters2d.push_back(*iNum);      
      }
    }
  }

  // Duplicate the pt parameters for some 2D histos
  for(int i =0; i < 2; i++){
    for (std::vector<double>::const_iterator iNum = thePhiParameters.begin();
         iNum != thePhiParameters.end();
         iNum++){

      if (iNum == thePhiParameters.begin()) {
        thePhiParameters2d.push_back(2*(*iNum));
      } else {
        thePhiParameters2d.push_back(*iNum);
      }

      if (i < 1){
        thePhiEtaParameters2d.push_back(*iNum);
      }
    }
  }

  //==========================================
  // Hard-coded parameters
  // Make modifibly from script later
  //==========================================

  theD0Parameters.push_back(50);
  theD0Parameters.push_back(-0.25);
  theD0Parameters.push_back(0.25);
  
  theZ0Parameters.push_back(50);
  theZ0Parameters.push_back(-25);
  theZ0Parameters.push_back(25);

  theChargeParameters.push_back(3);
  theChargeParameters.push_back(-1.5);
  theChargeParameters.push_back(1.5);

  theDRParameters.push_back(50);
  theDRParameters.push_back(0.0);
  theDRParameters.push_back(1.0);

  theChargeFlipParameters.push_back(2);
  theChargeFlipParameters.push_back(0.0);
  theChargeFlipParameters.push_back(1.0);

  //=======================================

  theMinPtCut    = pset.getUntrackedParameter<double>("MinPtCut");
  theMaxEtaCut   = pset.getUntrackedParameter<double>("MaxEtaCut");
  theL1DrCut     = pset.getUntrackedParameter<double>("L1DrCut");
  theL2DrCut     = pset.getUntrackedParameter<double>("L2DrCut");
  theL3DrCut     = pset.getUntrackedParameter<double>("L3DrCut");
  theMotherParticleId = pset.getUntrackedParameter<unsigned int> 
                        ("MotherParticleId");
  theNSigmas          = pset.getUntrackedParameter< std::vector<double> >
                        ("NSigmas90");

  theNtupleFileName = pset.getUntrackedParameter<std::string>
                      ( "NtupleFileName", "" );
  theNtuplePath     = pset.getUntrackedParameter<std::string>
                      ( "NtuplePath", "" );
  makeNtuple = false;
  if ( theTriggerName == theNtuplePath && theNtupleFileName != "" ) 
    makeNtuple = true;
  if ( makeNtuple ) {
    theFile      = new TFile(theNtupleFileName.c_str(),"RECREATE");
    TString vars = "eventNum:motherId:passL2Iso:passL3Iso:";
    vars        += "ptGen:etaGen:phiGen:";
    vars        += "ptL1:etaL1:phiL1:";
    vars        += "ptL2:etaL2:phiL2:";
    vars        += "ptL3:etaL3:phiL3:";
    for ( int i = 0; i < numCones; i++ ) {
      int cone  = (int)(coneSizes[i]*100);
      vars += Form("sumCaloIso%.2i:",cone);
      vars += Form("numCaloIso%.2i:",cone);
      vars += Form("sumEcalIso%.2i:",cone);
      vars += Form("sumHcalIso%.2i:",cone);
    }
    for ( int i = 0; i < numCones; i++ ) {
      int cone  = (int)(coneSizes[i]*100);
      for ( int j = 0; j < numMinPtCuts; j++ ) {
        int ptCut = (int)(minPtCuts[j]*10);
        vars += Form("sumTrackIso%.2i_%.2i:",ptCut,cone);
        vars += Form("numTrackIso%.2i_%.2i:",ptCut,cone);
      }
    }
    vars.Resize( vars.Length() - 1 );
    theNtuple    = new TNtuple("nt","data",vars);
  }

  dbe_ = 0 ;
  if ( pset.getUntrackedParameter<bool>("DQMStore", false) ) {
    dbe_ = Service<DQMStore>().operator->();
    dbe_->setVerbose(0);
  }

  eventNumber = 0;

  LogTrace ("HLTMuonVal") << "exiting constructor\n\n";

}



void HLTMuonGenericRate::finish()
{
  if ( makeNtuple ) {
    theFile->cd();
    theNtuple->Write();
    theFile->Close();
  }

  if (createStandAloneHistos && histoFileName != "") {
    dbe_->save(histoFileName);
  }
}



void HLTMuonGenericRate::analyze( const Event & iEvent )
{
  
  eventNumber++;
  LogTrace( "HLTMuonVal" ) << "\n\nIn analyze for trigger path " << 
    theTriggerName << ", Event:" << eventNumber <<"\n\n\n";

  // Update event numbers
  meNumberOfEvents->Fill(eventNumber); 

  //////////////////////////////////////////////////////////////////////////
  // Get all generated and reconstructed muons and create structs to hold  
  // matches to trigger candidates 

  double genMuonPt = -1;
  double recMuonPt = -1;


  LogTrace ("HLTMuonVal") << "\n\nStarting to look for gen muons\n\n";
                          
  
  std::vector<MatchStruct> genMatches;
  if ( useMuonFromGenerator ) {
    //   Handle<GenParticleCollection> genParticles;
    //     iEvent.getByLabel(theGenLabel, genParticles);
    //     for ( size_t i = 0; i < genParticles->size(); i++ ) {
    //       const reco::GenParticle *genParticle = &(*genParticles)[i];
    //       const Candidate *mother = findMother(genParticle);
    //       int    momId  = ( mother ) ? mother->pdgId() : 0;
    //       int    id     = genParticle->pdgId();
    //       int    status = genParticle->status();
    //       double pt     = genParticle->pt();
    //       double eta    = genParticle->eta();
    //       if ( abs(id) == 13  && status == 1 && 
    // 	   ( theMotherParticleId == 0 || abs(momId) == theMotherParticleId ) )
    //       {
    // 	MatchStruct newMatchStruct;
    // 	newMatchStruct.genCand = genParticle;
    // 	genMatches.push_back(newMatchStruct);
    // 	if ( pt > genMuonPt && fabs(eta) < theMaxEtaCut )
    // 	  genMuonPt = pt;
    //   } }
  }


  LogTrace ("HLTMuonVal") << "\n\n\n\nDone getting gen, now getting reco\n\n\n";
  
  std::vector<MatchStruct> recMatches;
  std::vector<MatchStruct> highPtMatches;
  reco::BeamSpot beamSpot;
  bool foundBeamSpot = false;
  
  if ( useMuonFromReco ) {
    Handle<reco::TrackCollection> muTracks;
    iEvent.getByLabel(theRecoLabel, muTracks);    
    reco::TrackCollection::const_iterator muon;
    if  ( muTracks.failedToGet() ) {
      LogWarning("HLTMuonVal") << "WARNING: failed to get the RECO Muon collection named " << theRecoLabel
                               << "\nYou have tracks to compare to... ignoring RECO muons"
                               << " for the rest of this job";
      useMuonFromReco = false;
    } else {
      for ( muon = muTracks->begin(); muon != muTracks->end(); ++muon ) {
        float pt  = muon->pt();
        float eta = muon->eta();
        MatchStruct newMatchStruct;
        newMatchStruct.recCand = &*muon;
        recMatches.push_back(newMatchStruct);

        LogTrace ("HLTMuonVal") << "\n\nFound a muon track in " << theRecoLabel
                                << " with pt = " << pt
                                << ", eta = " << eta;
        // Take out this eta cut, but still check to see if
        // it is a new maximum pt
        //if ( pt > recMuonPt  && fabs(eta) < theMaxEtaCut)
        if (pt > recMuonPt )
          recMuonPt = pt;
      }
    }

    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByLabel("offlineBeamSpot",recoBeamSpotHandle);
    if (!recoBeamSpotHandle.failedToGet()) {
      
      beamSpot = *recoBeamSpotHandle;
      foundBeamSpot = true;

      LogTrace ("HLTMuonVal") << "\n\n\nSUCESS finding beamspot\n\n\n" << endl;
      
    } else {
      LogWarning ("HLTMuonVal") << "FAILED to get the beamspot for this event";
    }

    // Now do the same Reco stuff, but try to use a new collection
    // that was produced by your muon selector
    //Handle<reco::TrackCollection> highPtTracks;
    
    // I am not 100% on what the modules are producing
    // but the collection name looks like the module name
    // and variations on the collection name don't seem to work
    // ie, name+"Tracks" doesn't give me what I want.

    // iEvent.getByLabel(highPtTrackCollection, highPtTracks);    

    //     reco::TrackCollection::const_iterator iHighPtTrack;
    //     if  ( highPtTracks.failedToGet() ) {
    //       LogWarning("HLTMuonVal") << "WARNING: failed to get the produced Muon collection ";

    //       //useMuonFromReco = false;
    //     } else {
    //       for ( iHighPtTrack = highPtTracks->begin(); iHighPtTrack != highPtTracks->end(); ++iHighPtTrack ) {
    //         float pt  = iHighPtTrack->pt();
    //         float eta = iHighPtTrack->eta();
    //         MatchStruct newMatchStruct;
    //         newMatchStruct.recCand = &*iHighPtTrack;
    //         highPtMatches.push_back(newMatchStruct);

        
    //         LogTrace ("HLTMuonVal") << "\n\nFound a PRODUCED reco muon with pt = " << pt
    //                                 << ", eta = " << eta;

        
    //         // Take out this eta cut, but still check to see if
    //         // it is a new maximum pt
    //         //if ( pt > recMuonPt  && fabs(eta) < theMaxEtaCut)
    //         //if (pt > recMuonPt )
    //         //  recMuonPt = pt;
    //       }
    //     }

    

  } 
  
  LogTrace("HLTMuonVal") << "\n\n\n\ngenMuonPt: " << genMuonPt << ", "  
                         << "recMuonPt: " << recMuonPt
                         << "\nNow preparing to get trigger objects" 
                         << "\n\n\n\n";

  //////////////////////////////////////////////////////////////////////////
  // Get the L1 and HLT trigger collections

  edm::Handle<trigger::TriggerEventWithRefs> rawTriggerEvent;
  edm::Handle<trigger::TriggerEvent>         aodTriggerEvent;
  vector<LorentzVector>                      l1Particles;
  vector<LorentzVector>                      l1RawParticles;
  //--  HLTParticles [0] is a vector of L2 matches
  //--  HLTParticles [1] is a vector of L1 matches

  // HLT particles are just 4 vectors
  vector< vector<LorentzVector> >            hltParticles(numHltLabels);

  // HLT cands are references to trigger objects
  vector< vector<RecoChargedCandidateRef> >  hltCands(numHltLabels);

  // L1 Cands are references to trigger objects
  vector<L1MuonParticleRef> l1Cands;
  
  InputTag collectionTag;
  size_t   filterIndex;


  //// Get the candidates from the RAW trigger summary

  //if ( !useAod ) {

  // Try to get the triggerSummaryRAW branch for
  // this event. If it's there, great, keep using it.
  // but if it isn't there, skip over it silently

  LogTrace ("HLTMuonVal") << "Trying to get RAW information\n\n";
                          
  iEvent.getByLabel( "hltTriggerSummaryRAW", rawTriggerEvent );
  
  if ( rawTriggerEvent.isValid() ) { 
    LogTrace("HLTMuonVal") << "\n\nRAW trigger summary found! "
                           << "\n\nUsing RAW information";
    
    collectionTag = InputTag( theL1CollectionLabel, "", theHltProcessName );
    filterIndex   = rawTriggerEvent->filterIndex(collectionTag);


    if ( filterIndex < rawTriggerEvent->size() ) {
      rawTriggerEvent->getObjects( filterIndex, TriggerL1Mu, l1Cands );
      LogTrace ("HLTMuonVal") << "Found l1 raw cands for filter = " << filterIndex ;                              
        
    } else {
      LogTrace("HLTMuonVal") << "No L1 Collection with label " 
                                << collectionTag;
    }
    
    //for ( size_t i = 0; i < l1Cands.size(); i++ ) 
    //  l1Cands.push_back( l1Cands[i]->p4() );
    LogTrace ("HLTMuonVal") << "Looking for information from  hltFilters";
                            
    for ( size_t i = 0; i < numHltLabels; i++ ) {

      collectionTag = InputTag( theHltCollectionLabels[i], 
                                "", theHltProcessName );
      filterIndex   = rawTriggerEvent->filterIndex(collectionTag);

      LogTrace ("HLTMuonVal") << "Looking for candidates for filter "
                              << theHltCollectionLabels[i]
                              << ", index = "
                              << filterIndex;
      
      if ( filterIndex < rawTriggerEvent->size() )
        rawTriggerEvent->getObjects( filterIndex, TriggerMuon, hltCands[i]);
      else LogTrace("HLTMuonVal") << "No HLT Collection with label " 
                                  << collectionTag;


      // don't copy the hltCands into particles
      // for ( size_t j = 0; j < hltCands[i].size(); j++ )
      // hltParticles[i].push_back( hltCands[i][j]->p4() );

    } // End loop over theHltCollectionLabels
  }  else {
    LogTrace ("HLTMuonVal") << "\n\nCouldn't find any RAW information for this event";
                            
  } // Done processing RAW summary information
    


  //// Get the candidates from the AOD trigger summary
  ///  JMS This is the unpacking that you might have
  ///  otherwise had to do 
  // if ( useAod ) {

    LogTrace ("HLTMuonVal") << "\n\n\nLooking for AOD branch named "
                            << "hltTriggerSummaryAOD\n\n\n";
                            
    iEvent.getByLabel("hltTriggerSummaryAOD", aodTriggerEvent);
    if ( !aodTriggerEvent.isValid() ) { 
      LogError("HLTMuonVal") << "No AOD trigger summary found! Returning..."; 
      return; 
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

      return;
    }

    // The AOD block has many collections, and you need to
    // parse which one you want. There are fancy lookup functions
    // to give you the number of the collection you want.
    // I think this is related to the trigger bits for each
    // event not being constant... so kinda like triger
    
    collectionTag = InputTag( theL1CollectionLabel, "", theHltProcessName );

    LogTrace ("HLTMuonVal") << "Trigger Name is " << theTriggerName;
    
    LogTrace ("HLTMuonVal") << "\n\n L1Collection tag is "
                            << collectionTag
                            << " and size filters is "
                            << aodTriggerEvent->sizeFilters()
                            << " Dumping full list of collection tags";

    LogTrace ("HLTMuonVal") << "\nL1LabelAodLabel = " << theAodL1Label << endl
                            << "\nL2LabelAodLabel = " << theAodL2Label << endl
                            << "\n\nLooping over L3 lables\n" ;


    //////////////////////////////////////////////////////
    // Print everything
    /////////////////////////////////////////////////////
    
    vector<string>::const_iterator iHltColl;
    int numHltColl = 0;
    for (  iHltColl = theHltCollectionLabels.begin();
           iHltColl != theHltCollectionLabels.end();
           iHltColl++ ) {
      LogTrace ("HLTMuonVal") << "Hlt label # "  << numHltColl
                              << " = "
                              << (*iHltColl);
      numHltColl++;
    }

    // Print out each collection that this event has
    vector<string> allAodCollTags = aodTriggerEvent->collectionTags();
    vector<string>::const_iterator iCollTag;
    int numColls = 0;
    for ( iCollTag = allAodCollTags.begin();
          iCollTag != allAodCollTags.end();
          iCollTag++ ) {
      
      LogTrace ("HLTMuonVal") << "Tag  " << numColls << " = "
                              << (*iCollTag) 
                              << endl;

      numColls++;
    }

    
    for ( size_t iFilter = 0; iFilter < aodTriggerEvent->sizeFilters(); iFilter++) {
      InputTag thisTag = aodTriggerEvent->filterTag(iFilter);
      LogTrace("HLTMuonVal") << "Filter number " << iFilter << "  tag = "
                             << thisTag << endl;
    }
    /////////////////////////////////////////////////////////////

    
    filterIndex   = aodTriggerEvent->filterIndex( collectionTag );

    LogTrace ("HLTMuonVal") << "\n\n filterIndex is "
                            << filterIndex;
    
    if ( filterIndex < aodTriggerEvent->sizeFilters() ) {
      const Keys &keys = aodTriggerEvent->filterKeys( filterIndex );

      LogTrace ("HLTMuonVal") << "\n\nGot keys";
      LogTrace ("HLTMuonVal") << "Key size is " << keys.size();
                              
      // The keys are apparently pointers into the trigger
      // objects collections
      // Use the key's to look up the particles for the
      // filter module that you're using 
      
      for ( size_t j = 0; j < keys.size(); j++ ){
        TriggerObject foundObject = objects[keys[j]];
        LogTrace ("HLTMuonVal") << "Storing a trigger object with id = "
                                << foundObject.id() << "\n\n";
        l1Particles.push_back( objects[keys[j]].particle().p4() );
      }
    } 
    ///////////////////////////////////////////////////////////////
    //     LogTrace ("HLTMuonVal") << "moving on to l2 collection";
    //     collectionTag = InputTag( theAodL2Label, "", theHltProcessName );
    //     filterIndex   = aodTriggerEvent->filterIndex( collectionTag );

    //     LogTrace ("HLTMuonVal") << "\n\n L2Collection tag is "
    //                             << collectionTag
    //                             << " and size filters is "
    //                             << aodTriggerEvent->sizeFilters();

    //     LogTrace ("HLTMuonVal") << "\n\n filterIndex is "
    //                             << filterIndex;
    
    //     if ( filterIndex < aodTriggerEvent->sizeFilters() ) {
      
    //       const Keys &keys = aodTriggerEvent->filterKeys( filterIndex );

    //       LogTrace ("HLTMuonVal") << "\n\nGot keys";
    //       LogTrace ("HLTMuonVal") << "Key size is " << keys.size();

    //       if (hltParticles.size() > 0) {
        
      
    //         for ( size_t j = 0; j < keys.size(); j++ ) {
    //           TriggerObject foundObject = objects[keys[j]];
    //           LogTrace ("HLTMuonVal") << "Storing a trigger object with id = "
    //                                 << foundObject.id() << "\n\n";

    //           hltParticles[0].push_back( objects[keys[j]].particle().p4() );

    //         }
    //       } else { // you don't have any hltLabels
    //         LogTrace ("HLTMuonVal") << "Oops, you don't have any hlt labels"
    //                                 << "but you do have l2 objects for this filter";
                                
    //       }
    //    } 
    ///////////////////////////////////////////////////////////////
    LogTrace ("HLTMuonVal") << "Moving onto L2 & L3";

    
    //if (theHltCollectionLabels.size() > 0) {
    int indexHltColl = 0;
    for (iHltColl = theHltCollectionLabels.begin();
         iHltColl != theHltCollectionLabels.end();
         iHltColl++ ){
      collectionTag = InputTag((*iHltColl) , "", 
                                theHltProcessName );
      filterIndex   = aodTriggerEvent->filterIndex( collectionTag );

      LogTrace ("HLTMuonVal") << "\n\n HLTCollection tag is "
                              << collectionTag
                              << " and size filters is "
                              << aodTriggerEvent->sizeFilters();
    
      LogTrace ("HLTMuonVal") << "\n\n filterIndex is "
                              << filterIndex;

    
      if ( filterIndex < aodTriggerEvent->sizeFilters() ) {
        const Keys &keys = aodTriggerEvent->filterKeys( filterIndex );
        for ( size_t j = 0; j < keys.size(); j++ ){
          TriggerObject foundObject = objects[keys[j]];
          LogTrace ("HLTMuonVal") << "Storing a trigger object with id = "
                                << foundObject.id() << "\n\n";

          hltParticles[indexHltColl].push_back( objects[keys[j]].particle().p4() );
        }
      }

      indexHltColl++;
    }
    
    // At this point, we should check whether the prescaled L1 and L2
    // triggers actually fired, and exit if not.
    // JMS -- does this still make sense to check?
    // JMS -- There is a chance that the L1/L2 collections
    //        won't be available, but that there is still an L3...
    //        this selection was unique to the AOD piece of the code.

    
    //if ( l1Particles.size() == 0 || hltParticles[0].size() == 0 ) 
    //   { LogTrace("HLTMuonVal") << "L1,L2  didn't fire, no trigger objects to compare to"; return; }
    
    //} // Done getting AOD trigger summary



  /////////////////////////////////////////////////////////////////////

  int totalNumOfHltParticles = 0;
  int tempIndexHltColl = 0;
  for ( vector<string>::const_iterator iHltColl = theHltCollectionLabels.begin();
        iHltColl != theHltCollectionLabels.end();
        iHltColl++ ){
    LogTrace ("HLTMuonVal") << "HLT label = " << (*iHltColl) 
                            << ", Number of hlt particles (4-vectors from aod) = "
                            << hltParticles[tempIndexHltColl].size()
                            << "\n";
    totalNumOfHltParticles += hltParticles[tempIndexHltColl].size();

    LogTrace ("HLTMuonVal") << "    Number of hlt cands (hltdebug refs) = " 
                            << hltCands[tempIndexHltColl].size()
                            << "\n";
    
    tempIndexHltColl++;
  }
  
  LogTrace ("HLTMuonVal") << "\n\nEvent " << eventNumber
                          << " has numL1Cands = " << l1Particles.size()
                          << " and numHltCands = " << totalNumOfHltParticles
                          << " now looking for matches\n\n" << endl;




  
  hNumObjects->getTH1()->AddBinContent( 3, l1Particles.size() );

  for ( size_t i = 0; i < numHltLabels; i++ ) 
    hNumObjects->getTH1()->AddBinContent( i + 4, hltParticles[i].size() );

  //////////////////////////////////////////////////////////////////////////
  // Initialize MatchStructs

  LorentzVector nullLorentzVector( 0., 0., 0., -999. );

  // a fake hlt cand is an hlt object not matched to a
  // reco object
  std::vector< std::vector<HltFakeStruct> > hltFakeCands(numHltLabels);

  for ( size_t i = 0; i < genMatches.size(); i++ ) {

    // this part of the code isn't maintained
    genMatches[i].l1Cand = nullLorentzVector;
    genMatches[i].hltCands. assign( numHltLabels, nullLorentzVector );
    genMatches[i].hltTracks.assign( numHltLabels, false );
  }

  for ( size_t i = 0; i < recMatches.size(); i++ ) {
    recMatches[i].l1Cand = nullLorentzVector;
    recMatches[i].hltCands. assign( numHltLabels, nullLorentzVector );
    recMatches[i].hltTracks.assign( numHltLabels, false );
    // new! raw matches too
    recMatches[i].hltRawCands.assign(numHltLabels, nullLorentzVector);
    recMatches[i].l1RawCand = nullLorentzVector;
  }




  
  //////////////////////////////////////////////////////////////////////////
  // Loop through L1 candidates, matching to gen/reco muons 

  unsigned int numL1Cands = 0;

  
  for ( size_t i = 0; i < l1Particles.size(); i++ ) {

    LorentzVector l1Cand = l1Particles[i];
    double eta           = l1Cand.eta();
    double phi           = l1Cand.phi();
    // L1 pt is taken from a lookup table
    // double ptLUT      = l1Cand->pt();  

    double maxDeltaR = theL1DrCut;
    numL1Cands++;

    if ( useMuonFromGenerator ){
      int match = findGenMatch( eta, phi, maxDeltaR, genMatches );

      // JMS why is this less than zero?
      if ( match != -1 && genMatches[match].l1Cand.E() < 0 ) {
        genMatches[match].l1Cand = l1Cand;
        LogTrace ("HLTMuonVal") << "Found a generator match to L1 cand";                              
      }
      else hNumOrphansGen->getTH1F()->AddBinContent( 1 );
    }

    if ( useMuonFromReco ){
      int match = findRecMatch( eta, phi, maxDeltaR, recMatches );
      if ( match != -1 && recMatches[match].l1Cand.E() < 0 ) {
        recMatches[match].l1Cand = l1Cand;
        LogTrace ("HLTMuonVal") << "Found a rec match to L1 particle (aod)";          
      } else {
        hNumOrphansRec->getTH1F()->AddBinContent( 1 );
      }
    }

  } // End loop over l1Particles

  ////////////////////////////////////////////////////////
  //   Loop over the L1 Candidates (RAW information)
  //   and look for matches
  ////////////////////////////////////////////////////////
  
  for ( size_t i = 0; i < l1Cands.size(); i++ ) {

    LorentzVector l1Cand = l1Cands[i]->p4();
    double eta           = l1Cand.eta();
    double phi           = l1Cand.phi();
    // L1 pt is taken from a lookup table
    // double ptLUT      = l1Cand->pt();  

    double maxDeltaR = theL1DrCut;
    //numL1Cands++;

    if ( useMuonFromGenerator ){
      int match = findGenMatch( eta, phi, maxDeltaR, genMatches );

      // If match didn't return fail, and we don't already
      // have an L1 cand
      if ( match != -1 && genMatches[match].l1Cand.E() < 0 ) {
        genMatches[match].l1Cand = l1Cand;
        LogTrace ("HLTMuonVal") << "Found a generator match to L1 cand";
        
      }
      else hNumOrphansGen->getTH1F()->AddBinContent( 1 );
    }

    if ( useMuonFromReco ){
      int match = findRecMatch( eta, phi, maxDeltaR, recMatches );
      if ( match != -1 && recMatches[match].l1RawCand.E() < 0 ) {
        recMatches[match].l1RawCand = l1Cand;
        LogTrace ("HLTMuonVal") << "Found an L1 match to a RAW object";
      } else {
        hNumOrphansRec->getTH1F()->AddBinContent( 1 );
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

      LorentzVector hltCand = hltParticles[i][candNum];
      double eta            = hltCand.eta();
      double phi            = hltCand.phi();

      numHltCands[i]++;

      if ( useMuonFromGenerator ){
        int match = findGenMatch( eta, phi, maxDeltaR, genMatches );
      
        if ( match != -1 && genMatches[match].hltCands[i].E() < 0 ) {
          genMatches[match].hltCands[i] = hltCand;
          LogTrace ("HLTMuonVal") << "Found a HLT cand match!";
          if ( !useAod ) genMatches[match].hltTracks[i] = 
                           &*hltCands[i][candNum];
        }
        else hNumOrphansGen->getTH1F()->AddBinContent( i + 2 );
      }

      if ( useMuonFromReco ){

        HltFakeStruct tempFakeCand; 
        tempFakeCand.myHltCand  = hltCand;

        int match  = findRecMatch( eta, phi, maxDeltaR, recMatches );

        // if match doesn't return error (-1)
        // and if this candidate spot isn't filled
        if ( match != -1 && recMatches[match].hltCands[i].E() < 0 ) {
          recMatches[match].hltCands[i] = hltCand;

          // since this matched, it's not a fake, so
          // record it as "not a fake"
          tempFakeCand.isAFake = false;

          
          // if match *did* return -1, then this is a fake  hlt candidate
          // it is fake because it isn't matched to a reco muon
          // 2009-03-24 oops, found a bug here, used to be != -1
          // fixed 
        } else if (match == -1){
          tempFakeCand.isAFake = true;
          hNumOrphansRec->getTH1F()->AddBinContent( i + 2 );
        }

        // add this cand 
        hltFakeCands[i].push_back(tempFakeCand);
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

      if ( useMuonFromGenerator ){
        int match = findGenMatch( eta, phi, maxDeltaR, genMatches );
      
        if ( match != -1 && genMatches[match].hltCands[i].E() < 0 ) {
          genMatches[match].hltCands[i] = hltCand;
          LogTrace ("HLTMuonVal") << "Found a HLT cand match!";
          if ( !useAod ) genMatches[match].hltTracks[i] = 
                           &*hltCands[i][candNum];
        }
        else hNumOrphansGen->getTH1F()->AddBinContent( i + 2 );
      }

      if ( useMuonFromReco ){

        //HltFakeStruct tempFakeCand; 
        //tempFakeCand.myHltCand  = hltCand;

        int match  = findRecMatch( eta, phi, maxDeltaR, recMatches );

        // if match doesn't return error (-1)
        // and if this candidate spot isn't filled
        if ( match != -1 && recMatches[match].hltCands[i].E() < 0 ) {
          recMatches[match].hltRawCands[i] = hltCand;
          LogTrace ("HLTMuonVal") << "Found a RAW hlt match to reco";
        }

        //else if (match == -1){
          //tempFakeCand.isAFake = true;
          //hNumOrphansRec->getTH1F()->AddBinContent( i + 2 );
          //}

        // add this cand 
        //hltFakeCands[i].push_back(tempFakeCand);
        //LogTrace ("HLTMuonVal") << "\n\nWas this a fake hlt cand? "
        //                      << tempFakeCand.isAFake;

      }

                              
      
      //LogTrace("HLTMuonVal") << "Number of HLT Cands: " << numHltCands[i];

    } // End loop over HLT RAW information


  } // End loop over HLT labels

  
  //////////////////////////////////////////////////////////////////////////
  // Fill ntuple

  if ( makeNtuple ) {
    Handle<reco::IsoDepositMap> caloDepMap, trackDepMap;
    iEvent.getByLabel("hltL2MuonIsolations",caloDepMap);
    iEvent.getByLabel("hltL3MuonIsolations",trackDepMap);
    IsoDeposit::Vetos vetos;
    if ( isIsolatedPath )
      for ( size_t i = 0; i < hltCands[2].size(); i++ ) {
	TrackRef tk = hltCands[2][i]->get<TrackRef>();
	vetos.push_back( (*trackDepMap)[tk].veto() );
      }
    for ( size_t i = 0; i < genMatches.size(); i++ ) {
      for ( int k = 0; k < 50; k++ ) theNtuplePars[k] = -99;

      //   ----------- Obsolete b/c it references GEN info
      //   ----------- You shouldn't be doing this 
      
      theNtuplePars[0] = eventNumber;      
      theNtuplePars[1] = -9e20; //(findMother(genMatches[i].genCand))->pdgId();
      theNtuplePars[4] = -9e20; //genMatches[i].genCand->pt();
      theNtuplePars[5] = -9e20; //genMatches[i].genCand->eta();
      theNtuplePars[6] = -9e20; //genMatches[i].genCand->phi();
      if ( genMatches[i].l1Cand.E() > 0 ) {
	theNtuplePars[7] = genMatches[i].l1Cand.pt();
	theNtuplePars[8] = genMatches[i].l1Cand.eta();
	theNtuplePars[9] = genMatches[i].l1Cand.phi();
      }
      for ( size_t j = 0; j < genMatches[i].hltCands.size(); j++ ) {
	if ( genMatches[i].hltCands[j].E() > 0 ) {
	  if ( j == 0 ) {
	    theNtuplePars[10] = genMatches[i].hltCands[j].pt();
	    theNtuplePars[11] = genMatches[i].hltCands[j].eta();
	    theNtuplePars[12] = genMatches[i].hltCands[j].phi();
	    if ( isIsolatedPath && !useAod ) {
	      TrackRef tk = genMatches[i].hltTracks[j]->get<TrackRef>();
	      const IsoDeposit &dep = (*caloDepMap)[tk];
	      for ( int m = 0; m < numCones; m++ ) {
		double dr = coneSizes[m];
		std::pair<double,int> depInfo = dep.depositAndCountWithin(dr);
		theNtuplePars[ 16 + 4*m + 0 ] = depInfo.first;
		theNtuplePars[ 16 + 4*m + 1 ] = depInfo.second;
	  } } }
	  if ( ( !isIsolatedPath && j == 1 ) ||
	       (  isIsolatedPath && j == 2 ) ) {
	    theNtuplePars[13] = genMatches[i].hltCands[j].pt();
	    theNtuplePars[14] = genMatches[i].hltCands[j].eta();
	    theNtuplePars[15] = genMatches[i].hltCands[j].phi();
	    if ( isIsolatedPath ) {
	      TrackRef tk = genMatches[i].hltTracks[j]->get<TrackRef>();
	      const IsoDeposit &dep = (*trackDepMap)[tk];
	      for ( int m = 0; m < numCones; m++ ) {
		for ( int n = 0; n < numMinPtCuts; n++ ) {
		  double dr = coneSizes[m];
		  double minPt = minPtCuts[n];
		  std::pair<double,int> depInfo;
		  depInfo = dep.depositAndCountWithin(dr, vetos, minPt);
		  int currentPlace = 16 + 4*numCones + 2*numMinPtCuts*m + 2*n;
		  theNtuplePars[ currentPlace + 0 ] = depInfo.first;
		  theNtuplePars[ currentPlace + 1 ] = depInfo.second;
		}
	  } } }
	  if ( isIsolatedPath && j == 1 ) theNtuplePars[2] = true;
	  if ( isIsolatedPath && j == 3 ) theNtuplePars[3] = true;
	}
      }
      theNtuple->Fill(theNtuplePars); 
    } // Done filling ntuple
  }
  
  //////////////////////////////////////////////////////////////////////////
  // Fill histograms

  // genMuonPt and recMuonPt are the max values
  // fill these hists with the max reconstructed Pt  
  if ( genMuonPt > 0 ) hPassMaxPtGen[0]->Fill( genMuonPt );
  if ( recMuonPt > 0 ) hPassMaxPtRec[0]->Fill( recMuonPt );

  int numRecMatches = recMatches.size();

  // there will be one hlt match for each
  // trigger module label
  // int numHltMatches = recMatches[i].hltCands.size();

  if (numRecMatches == 1) {
    if (recMuonPt >0) hPassExaclyOneMuonMaxPtRec[0]->Fill(recMuonPt);
  }

  // Fill these if there are any L1 candidates
  if ( numL1Cands >= theNumberOfObjects ) {
    if ( genMuonPt > 0 ) hPassMaxPtGen[1]->Fill( genMuonPt );
    if ( recMuonPt > 0 ) hPassMaxPtRec[1]->Fill( recMuonPt );
    if (numRecMatches == 1 && numL1Cands == 1) {
      if (recMuonPt >0) hPassExaclyOneMuonMaxPtRec[1]->Fill(recMuonPt);
    }
  }

  // Fill these if there are any L2/L3 candidates
  // Altered to only fill the hist if there is a match,
  // and not just a candidate
  // for ( size_t i = 0; i < numHltLabels; i++ ) {    
//     if ( numHltCands[i] >= theNumberOfObjects ) {
//       if ( genMuonPt > 0 ) hPassMaxPtGen[i+2]->Fill( genMuonPt );
//       if ( recMuonPt > 0 ) hPassMaxPtRec[i+2]->Fill( recMuonPt );
      
//       if (numRecMatches == 1 && numHltCands[i] == 1) {
//         if (recMuonPt >0) hPassExaclyOneMuonMaxPtRec[i+2]->Fill(recMuonPt);
//       }
//     }
    
//   }

  // Now actually do some matching
  //============= GEN ======================
  for ( size_t i = 0; i < genMatches.size(); i++ ) {
    double pt  = -9e20; //genMatches[i].genCand->pt();
    double eta = -9e20; //genMatches[i].genCand->eta();
    double phi = -9e20; //genMatches[i].genCand->phi();
    if ( pt > theMinPtCut &&  fabs(eta) < theMaxEtaCut ) {
      hNumObjects->getTH1()->AddBinContent(1);
      hPassEtaGen[0]->Fill(eta);
      hPassPhiGen[0]->Fill(phi);

      // if you found an L1 match, store it in L1 histos
      if ( genMatches[i].l1Cand.E() > 0 ) {
        hPassEtaGen[1]->Fill(eta);
        hPassPhiGen[1]->Fill(phi);
      }
      

      for ( size_t j = 0; j < genMatches[i].hltCands.size(); j++ ) {
        if ( genMatches[i].hltCands[j].E() > 0 ) {
          hPassEtaGen[j+2]->Fill(eta);
          hPassPhiGen[j+2]->Fill(phi);
        } 
      }
    }
  }
  
  ////////////////////////////////////////////
  //
  //               RECO Matching
  //
  ///////////////////////////////////////////

  double maxMatchPtRec = -10.0;
  //std::vector <double> allRecPts;
  //std::vector <bool> matchedToHLT;
  
  // Look at each rec & hlt cand
  for ( size_t i = 0; i < recMatches.size(); i++ ) {
    double pt  = recMatches[i].recCand->pt();
    double eta = recMatches[i].recCand->eta();
    double phi = recMatches[i].recCand->phi();

    //allRecPts.push_back(pt);

    // I think that these are measured w.r.t
    // (0,0,0)... you need to use other
    // functions to make them measured w.r.t
    // other locations
    
    double d0 = recMatches[i].recCand->d0();
    double z0 = recMatches[i].recCand->dz();
    double charge = recMatches[i].recCand->charge();

    double d0beam = -999;
    double z0beam = -999;
    
    if (foundBeamSpot) {
      d0beam = recMatches[i].recCand->dxy(beamSpot.position());
      z0beam = recMatches[i].recCand->dz(beamSpot.position());

      hBeamSpotZ0Rec[0]->Fill(beamSpot.z0());
    }
    
    // For now, take out the cuts on the pt/eta,
    // We'll get the total efficiency and worry about
    // the hlt matching later.    
    //    if ( pt > theMinPtCut &&  fabs(eta) < theMaxEtaCut ) {
    
    hNumObjects->getTH1()->AddBinContent(2);

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
    
    
    
    if (numRecMatches == 1) {
      hPassPtRecExactlyOne[0]->Fill(pt);
    }
    

    // if you found an L1 match, fill this histo
    if ( recMatches[i].l1Cand.E() > 0 ) {
      hPassEtaRec[1]->Fill(eta);
      hPassPhiRec[1]->Fill(phi);
      hPassPtRec[1]->Fill(pt);
      hPhiVsEtaRec[1]->Fill(eta,phi);
      hPassD0Rec[1]->Fill(d0);
      hPassD0BeamRec[1]->Fill(d0beam);
      hPassZ0Rec[1]->Fill(z0);
      hPassZ0BeamRec[1]->Fill(z0beam);
      hPassCharge[1]->Fill(charge);

      double l1eta = recMatches[i].l1Cand.eta();
      double l1phi = recMatches[i].l1Cand.phi();
      double deltaR = kinem::delta_R (l1eta, l1phi, eta, phi);
      
      
      hDeltaRMatched[1]->Fill(deltaR);
      //hChargeFlipMatched[1]->Fill();
      
      if (numRecMatches == 1) {
        hPassExaclyOneMuonMaxPtRec[1]->Fill(pt);
        hPassPtRecExactlyOne[1]->Fill(pt);
      }
    }
    
    //  bool foundAllPreviousCands = true;
    //  Look through the hltCands and see what's going on
    //

    
    for ( size_t j = 0; j < recMatches[i].hltCands.size(); j++ ) {
      if ( recMatches[i].hltCands[j].E() > 0 ) {
        double hltCand_pt = recMatches[i].hltCands[j].pt();
        double hltCand_eta = recMatches[i].hltCands[j].eta();
        double hltCand_phi = recMatches[i].hltCands[j].phi();

        // store this rec muon pt, not hlt cand pt
        if (theHltCollectionLabels.size() > j) {
          TString tempString = theHltCollectionLabels[j];
          if (tempString.Contains("L3")) {
            
            maxMatchPtRec = (pt > maxMatchPtRec)? pt : maxMatchPtRec;
          }
        }

        // these are histos where you have all, l1,l2,l3        
        hPassEtaRec[j+2]->Fill(eta);
        hPassPhiRec[j+2]->Fill(phi);
        hPassPtRec[j+2]->Fill(pt);
        hPhiVsEtaRec[j+2]->Fill(eta,phi);
        hPassD0Rec[j+2]->Fill(d0);
        hPassD0BeamRec[j+2]->Fill(d0beam);
        hPassZ0Rec[j+2]->Fill(z0);
        hPassZ0BeamRec[j+2]->Fill(z0beam);
        hPassCharge[j+2]->Fill(charge);

        
        // Histograms with Match in the name only have HLT
        // matches possible
        hPassMatchPtRec[j]->Fill(pt);
        hPtMatchVsPtRec[j]->Fill(hltCand_pt, pt);
        hEtaMatchVsEtaRec[j]->Fill(hltCand_eta, eta);
        hPhiMatchVsPhiRec[j]->Fill(hltCand_phi, phi);

        // Resolution histos must have hlt matches

        hResoPtAodRec[j]->Fill((pt - hltCand_pt)/pt);
        hResoEtaAodRec[j]->Fill((eta - hltCand_eta)/fabs(eta));
        hResoPhiAodRec[j]->Fill((phi - hltCand_phi)/fabs(phi));
        
        if (numRecMatches == 1 && (recMatches[i].hltCands.size()== 1)) {
          hPassExaclyOneMuonMaxPtRec[j+2]->Fill(pt);
          hPassPtRecExactlyOne[j+2]->Fill(pt);
        }
      }      
    }

    /////////////////////////////////////////////////
    //         Fill some RAW histograms
    /////////////////////////////////////////////////

    if ( recMatches[i].l1RawCand.E() > 0 ) {
      
      // you've found a L1 raw candidate
      rawMatchHltCandPt[1]->Fill(pt);
      rawMatchHltCandEta[1]->Fill(eta);
      rawMatchHltCandPhi[1]->Fill(phi);      
    }
    
    for ( size_t j = 0; j < recMatches[i].hltCands.size(); j++ ) {
      if ( recMatches[i].hltCands[j].E() > 0 ) {
        rawMatchHltCandPt[j+2]->Fill(pt);
        rawMatchHltCandEta[j+2]->Fill(eta);
        rawMatchHltCandPhi[j+2]->Fill(phi);   
      }
    }

    
  } // end RECO matching

  /////////////////////////////////////////
  //
  //  HLT fakes cands
  // 
  /////////////////////////////////////////


  for (unsigned int  iHltModule = 0;  iHltModule < numHltLabels; iHltModule++) {
    for(size_t iCand = 0; iCand < hltFakeCands[iHltModule].size() ; iCand ++){

      LorentzVector candVect = hltFakeCands[iHltModule][iCand].myHltCand;
      bool candIsFake = hltFakeCands[iHltModule][iCand].isAFake;
      
      allHltCandPt[iHltModule]->Fill(candVect.pt());
      allHltCandEta[iHltModule]->Fill(candVect.eta());
      allHltCandPhi[iHltModule]->Fill(candVect.phi());

      if (candIsFake) {
        fakeHltCandPt[iHltModule]->Fill(candVect.pt());
        fakeHltCandEta[iHltModule]->Fill(candVect.eta());
        fakeHltCandPhi[iHltModule]->Fill(candVect.phi());
        fakeHltCandEtaPhi[iHltModule]->Fill(candVect.eta(), candVect.phi());
      }
      
    }
    
  }
  

  LogTrace ("HLTMuonVal") << "There are " << recMatches.size()
                          << "  RECO muons in this event"
                          << endl;
    
  LogTrace ("HLTMuonVal") << "The max pt found by looking at candiates is   "
                          << maxMatchPtRec
                          << "\n and the max found while storing reco was "
                          << recMuonPt
                          << endl;
  
  for ( size_t i = 0; i < numHltLabels; i++ ) {
    // this will only fill up if L3
    // I don't think it's correct to fill
    // all the labels with this
    if (maxMatchPtRec > 0) hPassMaxPtRec[i+2]->Fill(maxMatchPtRec);
  }                                          
  

} // Done filling histograms



const reco::Candidate* HLTMuonGenericRate::
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

int HLTMuonGenericRate::findGenMatch
( double eta, double phi, double maxDeltaR, vector<MatchStruct> matches )
{
  double bestDeltaR = maxDeltaR;
  int bestMatch = -1;
  for ( size_t i = 0; i < matches.size(); i++ ) {
    // double dR = kinem::delta_R( eta, phi, 
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



int HLTMuonGenericRate::findRecMatch
( double eta, double phi,  double maxDeltaR, vector<MatchStruct> matches )
{
  double bestDeltaR = maxDeltaR;
  int bestMatch = -1;
  for ( size_t i = 0; i < matches.size(); i++ ) {
    double dR = kinem::delta_R( eta, phi, 
			        matches[i].recCand->eta(), 
				matches[i].recCand->phi() );
    if ( dR  < bestDeltaR ) {
      bestMatch  =  i;
      bestDeltaR = dR;
    }
  }
  return bestMatch;
}



void HLTMuonGenericRate::begin() 
{

  TString myLabel, newFolder;
  vector<TH1F*> h;

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
    newFolder = "HLT/Muon/Distributions/" + theTriggerName + "/" + theRecoLabel;

    
    
    dbe_->setCurrentFolder( newFolder.Data() );

    meNumberOfEvents            = dbe_->bookInt("NumberOfEvents");
    MonitorElement *meMinPtCut  = dbe_->bookFloat("MinPtCut"    );
    MonitorElement *meMaxEtaCut = dbe_->bookFloat("MaxEtaCut"   );
    meMinPtCut ->Fill(theMinPtCut );
    meMaxEtaCut->Fill(theMaxEtaCut);
    
    vector<string> binLabels;
    binLabels.push_back( theL1CollectionLabel.c_str() );
    for ( size_t i = 0; i < theHltCollectionLabels.size(); i++ )
      binLabels.push_back( theHltCollectionLabels[i].c_str() );

    hNumObjects = dbe_->book1D( "numObjects", "Number of Objects", 7, 0, 7 );
    hNumObjects->setBinLabel( 1, "Gen" );
    hNumObjects->setBinLabel( 2, "Reco" );
    for ( size_t i = 0; i < binLabels.size(); i++ )
      hNumObjects->setBinLabel( i + 3, binLabels[i].c_str() );
    hNumObjects->getTH1()->LabelsDeflate("X");

    if ( useMuonFromGenerator ){

      hNumOrphansGen = dbe_->book1D( "genNumOrphans", "Number of Orphans;;Number of Objects Not Matched to a Gen #mu", 5, 0, 5 );
      for ( size_t i = 0; i < binLabels.size(); i++ )
        hNumOrphansGen->setBinLabel( i + 1, binLabels[i].c_str() );
      hNumOrphansGen->getTH1()->LabelsDeflate("X");

      hPassMaxPtGen.push_back( bookIt( "genPassMaxPt_All", "pt of Leading Gen Muon", theMaxPtParameters) );
      hPassMaxPtGen.push_back( bookIt( "genPassMaxPt_" + myLabel, "pt of Leading Gen Muon, if matched to " + myLabel, theMaxPtParameters) );
      hPassEtaGen.push_back( bookIt( "genPassEta_All", "#eta of Gen Muons", theEtaParameters) );
      hPassEtaGen.push_back( bookIt( "genPassEta_" + myLabel, "#eta of Gen Muons matched to " + myLabel, theEtaParameters) );
      hPassPhiGen.push_back( bookIt( "genPassPhi_All", "#phi of Gen Muons", thePhiParameters) );
      hPassPhiGen.push_back( bookIt( "genPassPhi_" + myLabel, "#phi of Gen Muons matched to " + myLabel, thePhiParameters) );

    }

    if ( useMuonFromReco ){

      hNumOrphansRec = dbe_->book1D( "recNumOrphans", "Number of Orphans;;Number of Objects Not Matched to a Reconstructed #mu", 5, 0, 5 );
      for ( size_t i = 0; i < binLabels.size(); i++ )
        hNumOrphansRec->setBinLabel( i + 1, binLabels[i].c_str() );
      hNumOrphansRec->getTH1()->LabelsDeflate("X");


      
      
      // 0 = MaxPt_All
      hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_All", "pt of Leading Reco Muon" ,  theMaxPtParameters) );
      // 1 = MaxPt if matched to L1 Trigger
      hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_" + myLabel, "pt of Leading Reco Muon, if matched to " + myLabel,  theMaxPtParameters) );

      hPassEtaRec.push_back( bookIt( "recPassEta_All", "#eta of Reco Muons", theEtaParameters) );
      hPassEtaRec.push_back( bookIt( "recPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
      
      hPassPhiRec.push_back( bookIt( "recPassPhi_All", "#phi of Reco Muons", thePhiParameters) );
      hPassPhiRec.push_back( bookIt( "recPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );
      
      hPassPtRec.push_back( bookIt( "recPassPt_All", "Pt of  Reco Muon" ,  theMaxPtParameters) );
      hPassPtRec.push_back( bookIt( "recPassPt_" + myLabel, "pt  Reco Muon, if matched to " + myLabel,  theMaxPtParameters) );
      
      hPassPtRecExactlyOne.push_back( bookIt( "recPassPtExactlyOne_All", "pt of Leading Reco Muon (==1 muon)" ,  theMaxPtParameters) );
      hPassPtRecExactlyOne.push_back( bookIt( "recPassPtExactlyOne_" + myLabel, "pt of Leading Reco Muon (==1 muon), if matched to " + myLabel,  theMaxPtParameters) );
      
      hPassExaclyOneMuonMaxPtRec.push_back( bookIt("recPassExactlyOneMuonMaxPt_All", "pt of Leading Reco Muon in events with exactly one muon" ,  theMaxPtParameters) );
      hPassExaclyOneMuonMaxPtRec.push_back( bookIt("recPassExactlyOneMuonMaxPt_" + myLabel, "pt of Leading Reco Muon in events with exactly one muon match to " + myLabel ,  theMaxPtParameters) );

      hPassD0Rec.push_back( bookIt("recPassD0_All", "Track 2-D impact parameter wrt (0,0,0)(d0) ALL", theD0Parameters));
      hPassD0Rec.push_back( bookIt("recPassD0_" + myLabel, "Track 2-D impact parameter (0,0,0)(d0) " + myLabel, theD0Parameters));
      hPassD0BeamRec.push_back( bookIt("recPassD0Beam_All", "Track 2-D impact parameter wrt (beam)(d0) ALL", theD0Parameters));
      hPassD0BeamRec.push_back( bookIt("recPassD0Beam_" + myLabel, "Track 2-D impact parameter (beam)(d0) " + myLabel, theD0Parameters));
      
      hPassZ0Rec.push_back( bookIt("recPassZ0_All", "Track Z0 wrt (0,0,0) ALL", theZ0Parameters));
      hPassZ0Rec.push_back( bookIt("recPassZ0_" + myLabel, "Track Z0 (0,0,0) " + myLabel, theZ0Parameters));      
      hPassZ0BeamRec.push_back( bookIt("recPassZ0Beam_All", "Track Z0 wrt (beam) ALL", theZ0Parameters));
      hPassZ0BeamRec.push_back( bookIt("recPassZ0Beam_" + myLabel, "Track Z0 (beam) " + myLabel, theZ0Parameters));

      hPassCharge.push_back( bookIt("recPassCharge_All", "Track Charge  ALL", theChargeParameters));
      hPassCharge.push_back( bookIt("recPassCharge_" + myLabel, "Track Charge  " + myLabel, theChargeParameters));


      hBeamSpotZ0Rec.push_back ( bookIt("recBeamSpotZ0_All", "Z0 of beamspot for this event", theZ0Parameters));

      // these hisotgrams requite a match, and so will only have
      // L1,L2,L3 histograms... there will be zero entries in all
      // but keep it there for book keeping
      hDeltaRMatched.push_back ( bookIt("recDeltaRMatched_All" , "#Delta R between matched HLTCand", theDRParameters));
      hDeltaRMatched.push_back ( bookIt("recDeltaRMatched_" + myLabel, "#Delta R between matched HLTCand", theDRParameters));

      hChargeFlipMatched.push_back ( bookIt("recChargeFlipMatched_All" , "Charge Flip from hlt to RECO;HLT;Reco", theChargeFlipParameters)); 
      hChargeFlipMatched.push_back ( bookIt("recChargeFlipMatched_" + myLabel, "Charge Flip from hlt to RECO;HLT;Reco", theChargeFlipParameters)); 
      
      
      ////////////////////////////////////////////////
      //  RAW Histograms 
      ////////////////////////////////////////////////

      
      rawMatchHltCandPt.push_back( bookIt( "rawPassPt_All", "Pt of  Reco Muon" ,  theMaxPtParameters) );
      rawMatchHltCandPt.push_back( bookIt( "rawPassPt_" + myLabel, "pt  Reco Muon, if matched to " + myLabel,  theMaxPtParameters) );
      
      rawMatchHltCandEta.push_back( bookIt( "rawPassEta_All", "#eta of Reco Muons", theEtaParameters) );
      rawMatchHltCandEta.push_back( bookIt( "rawPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
      
      rawMatchHltCandPhi.push_back( bookIt( "rawPassPhi_All", "#phi of Reco Muons", thePhiParameters) );
      rawMatchHltCandPhi.push_back( bookIt( "rawPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );
      
      
      //=================================
      //          2-D Histograms
      //=================================
      
      hPhiVsEtaRec.push_back ( bookIt ("recPhiVsRecEta_All", "Reco #phi vs Reco #eta  ", thePhiEtaParameters2d));
      hPhiVsEtaRec.push_back ( bookIt ("recPhiVsRecEta_" + myLabel, "Reco #phi vs Reco #eta  " +myLabel, thePhiEtaParameters2d));

    }

    for ( unsigned int i = 0; i < theHltCollectionLabels.size(); i++ ) {

      myLabel = theHltCollectionLabels[i];
      TString level = ( myLabel.Contains("L2") ) ? "L2" : "L3";
      myLabel = myLabel(myLabel.Index(level),myLabel.Length());
      myLabel = myLabel(0,myLabel.Index("Filtered")+8);
      
      if ( useMuonFromGenerator ) {

        hPassMaxPtGen.push_back( bookIt( "genPassMaxPt_" + myLabel, "pt of Leading Gen Muon, if matched to " + myLabel, theMaxPtParameters) );   
        hPassEtaGen.push_back( bookIt( "genPassEta_" + myLabel, "#eta of Gen Muons matched to " + myLabel, theEtaParameters) );
        hPassPhiGen.push_back( bookIt( "genPassPhi_" + myLabel, "#phi of Gen Muons matched to " + myLabel, thePhiParameters) );

      }

      if ( useMuonFromReco ) {

        // These histos have All, L1, L2, L3
        hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_" + myLabel, "pt of Leading Reco Muon, if matched to " + myLabel, theMaxPtParameters) );     
        hPassEtaRec.push_back( bookIt( "recPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
        hPassPhiRec.push_back( bookIt( "recPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );                

        hPassPtRec.push_back ( bookIt( "recPassPt_" + myLabel, "Pt of  Reco Muon, if matched to " + myLabel, theMaxPtParameters) );
        hPassPtRecExactlyOne.push_back (bookIt( "recPassPtExactlyOne__" + myLabel, "pt of Leading Reco Muon (==1 muon), if matched to " + myLabel, theMaxPtParameters) );

        hPassExaclyOneMuonMaxPtRec.push_back( bookIt("recPassExactlyOneMuonMaxPt_" + myLabel, "pt of Leading Reco Muon in events with exactly one muon match to " + myLabel ,  theMaxPtParameters) );        
        hPhiVsEtaRec.push_back ( bookIt ("recPhiVsRecEta_" + myLabel, "Reco #phi vs Reco #eta  " +myLabel, thePhiEtaParameters2d));

        // Match histos only have numHltLabels indices
        hPassMatchPtRec.push_back( bookIt( "recPassMatchPt_" + myLabel, "Pt of Reco Muon that is matched to Trigger Muon " + myLabel, theMaxPtParameters) );

        hPtMatchVsPtRec.push_back (bookIt("recPtVsMatchPt" + myLabel, "Reco Pt vs Matched HLT Muon Pt" + myLabel ,  theMaxPtParameters2d) );
        hEtaMatchVsEtaRec.push_back( bookIt( "recEtaVsMatchEta_" + myLabel, "Reco #eta vs HLT #eta  " + myLabel, theEtaParameters2d) );
        hPhiMatchVsPhiRec.push_back( bookIt( "recPhiVsMatchPhi_" + myLabel, "Reco #phi vs HLT #phi  " + myLabel, thePhiParameters2d) );

        hResoPtAodRec.push_back ( bookIt ("recResoPt_" + myLabel, "TrigSumAOD to RECO P_T resolution", theResParameters));
        hResoEtaAodRec.push_back ( bookIt ("recResoEta_" + myLabel, "TrigSumAOD to RECO #eta resolution", theResParameters));
        hResoPhiAodRec.push_back ( bookIt ("recResoPhi_" + myLabel, "TrigSumAOD to RECO #phi resolution", theResParameters));

        hPassD0Rec.push_back( bookIt("recPassD0_" + myLabel, "Track 2-D impact parameter (Z0) " + myLabel, theD0Parameters));
        hPassD0BeamRec.push_back( bookIt("recPassD0Beam_" + myLabel, "Track 2-D impact parameter (beam)(d0) " + myLabel, theD0Parameters));
        hPassZ0Rec.push_back( bookIt("recPassZ0_" + myLabel, "Track Z0 " + myLabel, theZ0Parameters));
        hPassZ0BeamRec.push_back( bookIt("recPassZ0Beam_" + myLabel, "Track Z0 (0,0,0) " + myLabel, theZ0Parameters));
        hPassCharge.push_back( bookIt("recPassCharge_" + myLabel, "Track Charge  " + myLabel, theChargeParameters));

        hDeltaRMatched.push_back ( bookIt("recDeltaRMatched_" + myLabel, "#Delta R between matched HLTCand", theDRParameters));
        hChargeFlipMatched.push_back ( bookIt("recChargeFlipMatched_" + myLabel, "Charge Flip from hlt to RECO;HLT;Reco", theChargeFlipParameters)); 
        
        
        // these candidates are indexed by the number
        // of hlt labels
        allHltCandPt.push_back( bookIt("allHltCandPt_" + myLabel, "Pt of all HLT Muon Cands, for HLT " + myLabel, theMaxPtParameters));     
        allHltCandEta.push_back( bookIt("allHltCandEta_" + myLabel, "Eta of all HLT Muon Cands, for HLT " + myLabel, theEtaParameters));         
        allHltCandPhi.push_back( bookIt("allHltCandPhi_" + myLabel, "Phi of all HLT Muon Cands, for HLT " + myLabel, thePhiParameters));    

        fakeHltCandPt.push_back( bookIt("fakeHltCandPt_" + myLabel, "Pt of fake HLT Muon Cands, for HLT " + myLabel, theMaxPtParameters));     
        fakeHltCandEta.push_back( bookIt("fakeHltCandEta_" + myLabel, "Eta of fake HLT Muon Cands, for HLT " + myLabel, theEtaParameters));         
        fakeHltCandPhi.push_back( bookIt("fakeHltCandPhi_" + myLabel, "Phi of fake HLT Muon Cands, for HLT " + myLabel, thePhiParameters));    
                
        fakeHltCandEtaPhi.push_back(bookIt("fakeHltCandPhiVsEta_" + myLabel, " AOD #phi vs  #eta for fake HLT Muon Cands, for HLT  " +myLabel, thePhiEtaParameters2d));

        // raw histograms

        rawMatchHltCandPt.push_back( bookIt( "rawPassPt_" + myLabel, "pt  Reco Muon, if matched to " + myLabel,  theMaxPtParameters) );
        rawMatchHltCandEta.push_back( bookIt( "rawPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
        rawMatchHltCandPhi.push_back( bookIt( "rawPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );
        
      }

    }
  }

}



MonitorElement* HLTMuonGenericRate::bookIt
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
    return dbe_->book1D( name.Data(), h );
    delete h;

    // this is the case for a 2D hist
  } else if (parameters.size() == 6) {

    int nBins2  = (int)parameters[3];
    double min2 = parameters[4];
    double max2 = parameters[5];

    TH2F *h = new TH2F (name, title, nBins, min, max, nBins2, min2, max2);
    h->Sumw2();
    return dbe_->book2D (name.Data(), h);
    delete h;

  } else {
    LogError ("HLTMuonVal") << "Directory" << dbe_->pwd() << " Name "
                            << name << " had an invalid number of paramters";
    return 0;
  }
  
}


