 /** \file HLTMuonGenericRate.cc
 *  Get L1/HLT efficiency/rate plots
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2009/02/17 15:15:57 $
 *  $Revision: 1.61 $
 */


#include "HLTriggerOffline/Muon/interface/HLTMuonGenericRate.h"
#include "HLTriggerOffline/Muon/interface/AnglesUtil.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

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
( const ParameterSet& pset, string triggerName, vector<string> moduleNames )
{

  theHltProcessName  = pset.getParameter<string>("HltProcessName");
  theNumberOfObjects = ( TString(triggerName).Contains("Double") ) ? 2 : 1;
  theTriggerName     = triggerName;

  useAod         = pset.getUntrackedParameter<bool>("UseAod");
  theAodL1Label  = pset.getUntrackedParameter<string>("AodL1Label");
  theAodL2Label  = pset.getUntrackedParameter<string>("AodL2Label");

  theHltCollectionLabels.clear();
  TPRegexp l1Regexp("L1.*Filtered");
  for ( size_t i = 0; i < moduleNames.size(); i++ ) {
    string module = moduleNames[i];
    if ( TString(module).Contains(l1Regexp) ) 
      theL1CollectionLabel = module;
    else if ( TString(module).Contains("Filtered") ) 
      theHltCollectionLabels.push_back(module);
  }
  if ( useAod ) {
    theL1CollectionLabel = theAodL1Label;
    string finalLabel    = theHltCollectionLabels.back();
    theHltCollectionLabels.clear();
    theHltCollectionLabels.push_back( theAodL2Label );
    theHltCollectionLabels.push_back( finalLabel );    
  }

  numHltLabels   = theHltCollectionLabels.size();
  isIsolatedPath = ( numHltLabels == 4 ) ? true : false;

  theGenLabel          = pset.getUntrackedParameter<string>("GenLabel" ,"");
  theRecoLabel         = pset.getUntrackedParameter<string>("RecoLabel","");  
  useMuonFromGenerator = ( theGenLabel  == "" ) ? false : true;
  useMuonFromReco      = ( theRecoLabel == "" ) ? false : true;

  theMaxPtParameters = pset.getParameter< vector<double> >("MaxPtParameters");
  thePtParameters    = pset.getParameter< vector<double> >("PtParameters");
  theEtaParameters   = pset.getParameter< vector<double> >("EtaParameters");
  thePhiParameters   = pset.getParameter< vector<double> >("PhiParameters");

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

}



void HLTMuonGenericRate::finish()
{
  if ( makeNtuple ) {
    theFile->cd();
    theNtuple->Write();
    theFile->Close();
  }
}



void HLTMuonGenericRate::analyze( const Event & iEvent )
{
  
  eventNumber++;
  LogTrace( "HLTMuonVal" ) << "In analyze for trigger path " << 
    theTriggerName << ", Event:" << eventNumber;

  // Update event numbers
  meNumberOfEvents->Fill(eventNumber); 

  //////////////////////////////////////////////////////////////////////////
  // Get all generated and reconstructed muons and create structs to hold  
  // matches to trigger candidates 

  double genMuonPt = -1;
  double recMuonPt = -1;

  std::vector<MatchStruct> genMatches;
  if ( useMuonFromGenerator ) {
    Handle<GenParticleCollection> genParticles;
    iEvent.getByLabel(theGenLabel, genParticles);
    for ( size_t i = 0; i < genParticles->size(); i++ ) {
      const reco::GenParticle *genParticle = &(*genParticles)[i];
      const Candidate *mother = findMother(genParticle);
      int    momId  = ( mother ) ? mother->pdgId() : 0;
      int    id     = genParticle->pdgId();
      int    status = genParticle->status();
      double pt     = genParticle->pt();
      double eta    = genParticle->eta();
      if ( abs(id) == 13  && status == 1 && 
	   ( theMotherParticleId == 0 || abs(momId) == theMotherParticleId ) )
      {
	MatchStruct newMatchStruct;
	newMatchStruct.genCand = genParticle;
	genMatches.push_back(newMatchStruct);
	if ( pt > genMuonPt && fabs(eta) < theMaxEtaCut )
	  genMuonPt = pt;
  } } }

  std::vector<MatchStruct> recMatches;
  if ( useMuonFromReco ) {
    Handle<reco::TrackCollection> muTracks;
    iEvent.getByLabel(theRecoLabel, muTracks);    
    reco::TrackCollection::const_iterator muon;
    if  ( muTracks.failedToGet() ) {
      LogTrace("HLTMuonVal") << "No reco tracks to compare to";
      useMuonFromReco = false;
    } else {
      for ( muon = muTracks->begin(); muon != muTracks->end(); ++muon ) {
	float pt  = muon->pt();
	float eta = muon->eta();
	MatchStruct newMatchStruct;
	newMatchStruct.recCand = &*muon;
	recMatches.push_back(newMatchStruct);
	if ( pt > recMuonPt && fabs(eta) < theMaxEtaCut ) 
	  recMuonPt = pt;
  } } } 
  
  LogTrace("HLTMuonVal") << "genMuonPt: " << genMuonPt << ", "  
                         << "recMuonPt: " << recMuonPt;

  //////////////////////////////////////////////////////////////////////////
  // Get the L1 and HLT trigger collections

  edm::Handle<trigger::TriggerEventWithRefs> rawTriggerEvent;
  edm::Handle<trigger::TriggerEvent>         aodTriggerEvent;
  vector<LorentzVector>                      l1Particles;
  vector< vector<LorentzVector> >            hltParticles(numHltLabels);
  vector< vector<RecoChargedCandidateRef> >  hltCands(numHltLabels);
  InputTag collectionTag;
  size_t   filterIndex;


  //// Get the candidates from the RAW trigger summary

  if ( !useAod ) {

    iEvent.getByLabel( "hltTriggerSummaryRAW", rawTriggerEvent );
    if ( !rawTriggerEvent.isValid() ) { 
      LogError("HLTMuonVal") << "No RAW trigger summary found! "
			     << "Change UseAod to ""True"" in configuration";
      return;
    }

    collectionTag = InputTag( theL1CollectionLabel, "", theHltProcessName );
    filterIndex   = rawTriggerEvent->filterIndex(collectionTag);
    vector<L1MuonParticleRef> l1Cands;

    if ( filterIndex < rawTriggerEvent->size() ) 
      rawTriggerEvent->getObjects( filterIndex, TriggerL1Mu, l1Cands );
    else LogTrace("HLTMuonVal") << "No L1 Collection with label " 
				<< collectionTag;
    
    for ( size_t i = 0; i < l1Cands.size(); i++ ) 
      l1Particles.push_back( l1Cands[i]->p4() );
    
    for ( size_t i = 0; i < numHltLabels; i++ ) {

      collectionTag = InputTag( theHltCollectionLabels[i], 
				"", theHltProcessName );
      filterIndex   = rawTriggerEvent->filterIndex(collectionTag);

      if ( filterIndex < rawTriggerEvent->size() )
	rawTriggerEvent->getObjects( filterIndex, TriggerMuon, hltCands[i]);
      else LogTrace("HLTMuonVal") << "No HLT Collection with label " 
				  << collectionTag;

      for ( size_t j = 0; j < hltCands[i].size(); j++ )
	hltParticles[i].push_back( hltCands[i][j]->p4() );

    } // End loop over theHltCollectionLabels

  } // Done getting RAW trigger summary


  //// Get the candidates from the AOD trigger summary

  if ( useAod ) {

    iEvent.getByLabel("hltTriggerSummaryAOD", aodTriggerEvent);
    if ( !aodTriggerEvent.isValid() ) { 
      LogError("HLTMuonVal") << "No AOD trigger summary found! Returning..."; 
      return; 
    }

    const TriggerObjectCollection objects = aodTriggerEvent->getObjects();

    collectionTag = InputTag( theAodL1Label, "", theHltProcessName );
    filterIndex   = aodTriggerEvent->filterIndex( collectionTag );
    if ( filterIndex < aodTriggerEvent->sizeFilters() ) {
      const Keys &keys = aodTriggerEvent->filterKeys( filterIndex );
      for ( size_t j = 0; j < keys.size(); j++ )
	l1Particles.push_back( objects[keys[j]].particle().p4() );
    } 

    collectionTag = InputTag( theAodL2Label, "", theHltProcessName );
    filterIndex   = aodTriggerEvent->filterIndex( collectionTag );
    if ( filterIndex < aodTriggerEvent->sizeFilters() ) {
      const Keys &keys = aodTriggerEvent->filterKeys( filterIndex );
      for ( size_t j = 0; j < keys.size(); j++ )
	hltParticles[0].push_back( objects[keys[j]].particle().p4() );
    } 

    collectionTag = InputTag( theHltCollectionLabels.back(), "", 
			      theHltProcessName );
    filterIndex   = aodTriggerEvent->filterIndex( collectionTag );
    if ( filterIndex < aodTriggerEvent->sizeFilters() ) {
      const Keys &keys = aodTriggerEvent->filterKeys( filterIndex );
      for ( size_t j = 0; j < keys.size(); j++ )
	hltParticles[1].push_back( objects[keys[j]].particle().p4() );
    } 

    // At this point, we should check whether the prescaled L1 and L2
    // triggers actually fired, and exit if not.
    if ( l1Particles.size() == 0 || hltParticles[0].size() == 0 ) 
      { LogTrace("HLTMuonVal") << "L1 or L2 didn't fire"; return; }
    
  } // Done getting AOD trigger summary
  
  hNumObjects->getTH1()->AddBinContent( 3, l1Particles.size() );

  for ( size_t i = 0; i < numHltLabels; i++ ) 
    hNumObjects->getTH1()->AddBinContent( i + 4, hltParticles[i].size() );

  //////////////////////////////////////////////////////////////////////////
  // Initialize MatchStructs

  LorentzVector nullLorentzVector( 0., 0., 0., -999. );

  for ( size_t i = 0; i < genMatches.size(); i++ ) {
    genMatches[i].l1Cand = nullLorentzVector;
    genMatches[i].hltCands. assign( numHltLabels, nullLorentzVector );
    genMatches[i].hltTracks.assign( numHltLabels, false );
  }

  for ( size_t i = 0; i < recMatches.size(); i++ ) {
    recMatches[i].l1Cand = nullLorentzVector;
    recMatches[i].hltCands. assign( numHltLabels, nullLorentzVector );
    recMatches[i].hltTracks.assign( numHltLabels, false );
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
      if ( match != -1 && genMatches[match].l1Cand.E() < 0 ) 
	genMatches[match].l1Cand = l1Cand;
      else hNumOrphansGen->getTH1F()->AddBinContent( 1 );
    }

    if ( useMuonFromReco ){
      int match = findRecMatch( eta, phi, maxDeltaR, recMatches );
      if ( match != -1 && recMatches[match].l1Cand.E() < 0 ) 
	recMatches[match].l1Cand = l1Cand;
      else hNumOrphansRec->getTH1F()->AddBinContent( 1 );
    }

  } // End loop over l1Particles

  LogTrace("HLTMuonVal") << "Number of L1 Cands: " << numL1Cands;

  //////////////////////////////////////////////////////////////////////////
  // Loop through HLT candidates, matching to gen/reco muons

  vector<unsigned int> numHltCands( numHltLabels, 0) ;

  for ( size_t i = 0; i < numHltLabels; i++ ) { 

    int triggerLevel      = ( i < ( numHltLabels / 2 ) ) ? 2 : 3;
    double maxDeltaR      = ( triggerLevel == 2 ) ? theL2DrCut : theL3DrCut;

    for ( size_t candNum = 0; candNum < hltParticles[i].size(); candNum++ ) {

      LorentzVector hltCand = hltParticles[i][candNum];
      double eta            = hltCand.eta();
      double phi            = hltCand.phi();

      numHltCands[i]++;

      if ( useMuonFromGenerator ){
	int match = findGenMatch( eta, phi, maxDeltaR, genMatches );
      
	if ( match != -1 && genMatches[match].hltCands[i].E() < 0 ) {
	  genMatches[match].hltCands[i] = hltCand;
	  if ( !useAod ) genMatches[match].hltTracks[i] = 
	     &*hltCands[i][candNum];
	}
	else hNumOrphansGen->getTH1F()->AddBinContent( i + 2 );
      }

      if ( useMuonFromReco ){
	int match  = findRecMatch( eta, phi, maxDeltaR, recMatches );
	if ( match != -1 && recMatches[match].hltCands[i].E() < 0 ) 
	  recMatches[match].hltCands[i] = hltCand;
	else hNumOrphansRec->getTH1F()->AddBinContent( i + 2 );
      }

      LogTrace("HLTMuonVal") << "Number of L1 Cands: " << numHltCands[i];

    } // End loop over HLT particles

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
      theNtuplePars[0] = eventNumber;
      theNtuplePars[1] = (findMother(genMatches[i].genCand))->pdgId();
      theNtuplePars[4] = genMatches[i].genCand->pt();
      theNtuplePars[5] = genMatches[i].genCand->eta();
      theNtuplePars[6] = genMatches[i].genCand->phi();
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

  if ( genMuonPt > 0 ) hPassMaxPtGen[0]->Fill( genMuonPt );
  if ( recMuonPt > 0 ) hPassMaxPtRec[0]->Fill( recMuonPt );
  if ( numL1Cands >= theNumberOfObjects ) {
    if ( genMuonPt > 0 ) hPassMaxPtGen[1]->Fill( genMuonPt );
    if ( recMuonPt > 0 ) hPassMaxPtRec[1]->Fill( recMuonPt );
  }
  for ( size_t i = 0; i < numHltLabels; i++ ) {
    if ( numHltCands[i] >= theNumberOfObjects ) {
      if ( genMuonPt > 0 ) hPassMaxPtGen[i+2]->Fill( genMuonPt );
      if ( recMuonPt > 0 ) hPassMaxPtRec[i+2]->Fill( recMuonPt );
    }
  }

  for ( size_t i = 0; i < genMatches.size(); i++ ) {
    double pt  = genMatches[i].genCand->pt();
    double eta = genMatches[i].genCand->eta();
    double phi = genMatches[i].genCand->phi();
    if ( pt > theMinPtCut &&  fabs(eta) < theMaxEtaCut ) {
      hNumObjects->getTH1()->AddBinContent(1);
      hPassEtaGen[0]->Fill(eta);
      hPassPhiGen[0]->Fill(phi);
      if ( genMatches[i].l1Cand.E() > 0 ) {
	hPassEtaGen[1]->Fill(eta);
	hPassPhiGen[1]->Fill(phi);
	bool foundAllPreviousCands = true;
	for ( size_t j = 0; j < genMatches[i].hltCands.size(); j++ ) {
	  if ( foundAllPreviousCands && genMatches[i].hltCands[j].E() > 0 ) {
	    hPassEtaGen[j+2]->Fill(eta);
	    hPassPhiGen[j+2]->Fill(phi);
	  } else foundAllPreviousCands = false;
  } } } }

  for ( size_t i = 0; i < recMatches.size(); i++ ) {
    double pt  = recMatches[i].recCand->pt();
    double eta = recMatches[i].recCand->eta();
    double phi = recMatches[i].recCand->phi();
    if ( pt > theMinPtCut &&  fabs(eta) < theMaxEtaCut ) {
      hNumObjects->getTH1()->AddBinContent(2);
      hPassEtaRec[0]->Fill(eta);
      hPassPhiRec[0]->Fill(phi);
      if ( recMatches[i].l1Cand.E() > 0 ) {
	hPassEtaRec[1]->Fill(eta);
	hPassPhiRec[1]->Fill(phi);
	bool foundAllPreviousCands = true;
	for ( size_t j = 0; j < recMatches[i].hltCands.size(); j++ ) {
	  if ( foundAllPreviousCands && recMatches[i].hltCands[j].E() > 0 ) {
	    hPassEtaRec[j+2]->Fill(eta);
	    hPassPhiRec[j+2]->Fill(phi);
	  } else foundAllPreviousCands = false;
  } } } }

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



int HLTMuonGenericRate::findGenMatch
( double eta, double phi, double maxDeltaR, vector<MatchStruct> matches )
{
  double bestDeltaR = maxDeltaR;
  int bestMatch = -1;
  for ( size_t i = 0; i < matches.size(); i++ ) {
    double dR = kinem::delta_R( eta, phi, 
				matches[i].genCand->eta(), 
				matches[i].genCand->phi() );
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

    myLabel = theL1CollectionLabel;
    myLabel = myLabel(myLabel.Index("L1"),myLabel.Length());
    myLabel = myLabel(0,myLabel.Index("Filtered")+8);

    newFolder = "HLT/Muon/Distributions/" + theTriggerName;
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

      hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_All", "pt of Leading Reco Muon" + myLabel,  theMaxPtParameters) );
      hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_" + myLabel, "pt of Leading Reco Muon, if matched to " + myLabel,  theMaxPtParameters) );
      hPassEtaRec.push_back( bookIt( "recPassEta_All", "#eta of Reco Muons", theEtaParameters) );
      hPassEtaRec.push_back( bookIt( "recPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
      hPassPhiRec.push_back( bookIt( "recPassPhi_All", "#phi of Reco Muons", thePhiParameters) );
      hPassPhiRec.push_back( bookIt( "recPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );

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

	hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_" + myLabel, "pt of Leading Reco Muon, if matched to " + myLabel, theMaxPtParameters) );     
	hPassEtaRec.push_back( bookIt( "recPassEta_" + myLabel, "#eta of Reco Muons matched to " + myLabel, theEtaParameters) );
	hPassPhiRec.push_back( bookIt( "recPassPhi_" + myLabel, "#phi of Reco Muons matched to " + myLabel, thePhiParameters) );
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
  TH1F *h = new TH1F( name, title, nBins, min, max );
  h->Sumw2();
  return dbe_->book1D( name.Data(), h );
  delete h;
}

