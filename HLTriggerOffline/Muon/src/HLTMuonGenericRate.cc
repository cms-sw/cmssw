 /** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author J. Klukas, M. Vander Donckt (copied from J. Alcaraz)
 */

//// Documentation available on the CMS TWiki:
//// https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance

#include "HLTriggerOffline/Muon/interface/HLTMuonGenericRate.h"
#include "HLTriggerOffline/Muon/interface/AnglesUtil.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

// For storing calorimeter isolation info in the ntuple
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include <iostream>

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;

typedef std::vector< edm::ParameterSet > Parameters;

const int numCones = 5;
double coneSizes[] = { 0.20, 0.24, 0.30, 0.35, 0.40 };
const int numMinPtCuts = 5;
double minPtCuts[] = { 0., 1.5, 2., 3., 5. };


/// Constructor
HLTMuonGenericRate::HLTMuonGenericRate( const ParameterSet& pset, 
					string triggerName,
					vector<string> moduleNames )
{

  theHltProcessName  = pset.getParameter<string>("HltProcessName");
  theNumberOfObjects = ( TString(triggerName).Contains("Double") ) ? 2 : 1;
  theTriggerName     = triggerName;

  theHltCollectionLabels.clear();
  for ( size_t i = 0; i < moduleNames.size(); i++ ) {
    string module = moduleNames[i];
    if ( TString(module).Contains("L1Filtered") ) 
      theL1CollectionLabel = module;
    else if ( TString(module).Contains("Filtered") ) 
      theHltCollectionLabels.push_back(module);
  }

  m_useMuonFromGenerator = pset.getParameter<bool>("UseMuonFromGenerator");
  m_useMuonFromReco      = pset.getParameter<bool>("UseMuonFromReco");
  if ( m_useMuonFromGenerator ) 
    theGenLabel          = pset.getUntrackedParameter<string>("GenLabel");
  if ( m_useMuonFromReco )
    theRecoLabel         = pset.getUntrackedParameter<string>("RecoLabel");

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
  m_makeNtuple = false;
  if ( theTriggerName == theNtuplePath && theNtupleFileName != "" ) 
    m_makeNtuple = true;
  if ( m_makeNtuple ) {
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
      for ( int j = 0; j < numMinPtCuts; j++ ) {
	int ptCut = (int)(minPtCuts[j]*10);
	vars += Form("sumTrackIso%.2i_%.2i:",ptCut,cone);
	vars += Form("sumTrackIso%.2i_%.2i:",ptCut,cone);
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

  theEventNumber     = 0;

}



void HLTMuonGenericRate::finish()
{
  if ( m_makeNtuple ) {
    theFile->cd();
    theNtuple->Write();
    theFile->Write();
    theFile->Close();
  }
}



void HLTMuonGenericRate::analyze( const Event & iEvent )
{

  theEventNumber++;
  LogTrace( "HLTMuonVal" ) << "In analyze for L1 trigger " << 
    theL1CollectionLabel << " Event:" << theEventNumber;  

  // Update event numbers
  NumberOfEvents  ->Fill(theEventNumber     ); 

  //////////////////////////////////////////////////////////////////////////
  // Get all generated and reconstructed muons and create structs to hold  
  // matches to trigger candidates 

  double genMuonPt = -1;
  double recMuonPt = -1;

  std::vector<MatchStruct> genMatches;
  if ( m_useMuonFromGenerator ) {
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
  if ( m_useMuonFromReco ) {
    Handle<reco::TrackCollection> muTracks;
    iEvent.getByLabel(theRecoLabel, muTracks);    
    reco::TrackCollection::const_iterator muon;
    if  ( muTracks.failedToGet() ) {
      LogDebug("HLTMuonVal") << "No reco tracks to compare to";
      m_useMuonFromReco = false;
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
  
  if ( genMuonPt > 0 ) hPassMaxPtGen[0]->Fill(genMuonPt);
  if ( recMuonPt > 0 ) hPassMaxPtRec[0]->Fill(recMuonPt);

  LogTrace("HLTMuonVal") << "genMuonPt: " << genMuonPt << ", "  
                         << "recMuonPt: " << recMuonPt;

  //////////////////////////////////////////////////////////////////////////
  // Get the L1 and HLT trigger collections

  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  iEvent.getByLabel("hltTriggerSummaryRAW", triggerObj); 

  if( !triggerObj.isValid() ) { 
    LogDebug("HLTMuonVal") << "RAW-type HLT results not found, skipping event";
    return;
  }

  // Output filter information

  const size_type numFilterObjects(triggerObj->size());
  LogTrace("HLTMuonVal") 
    << "Used Processname: " << triggerObj->usedProcessName() << "\n"
    << "Number of TriggerFilterObjects: " << numFilterObjects << "\n"
    << "The TriggerFilterObjects: #, tag";
  for ( size_type i = 0; i != numFilterObjects; ++i ) LogTrace("HLTMuonVal") 
    << i << " " << triggerObj->filterTag(i).encode();

  InputTag tag; // Used as a shortcut for the current CollectionLabel

  // Get the L1 candidates //

  vector<L1MuonParticleRef> l1Cands;
  tag = InputTag(theL1CollectionLabel,"",theHltProcessName);
  LogTrace("HLTMuonVal") << "TriggerObject Size=" << triggerObj->size();

  if ( triggerObj->filterIndex(tag) >= triggerObj->size() ) {
    LogTrace("HLTMuonVal") << "No L1 Collection with label " << tag;
    return;
  } else {
    size_t filterIndex = triggerObj->filterIndex(tag);
    triggerObj->getObjects( filterIndex, TriggerL1Mu, l1Cands );
  }

  hNumObjects->getTH1()->AddBinContent( 3, l1Cands.size() );

  // Get the HLT candidates //

  unsigned int numHltLabels = theHltCollectionLabels.size();
  bool isIsolatedPath = ( numHltLabels == 4 ) ? true : false;
  vector< vector<RecoChargedCandidateRef> > hltCands(numHltLabels);

  for ( unsigned int i = 0; i < numHltLabels; i++ ) {

    tag = InputTag(theHltCollectionLabels[i],"",theHltProcessName);

    if ( triggerObj->filterIndex(tag) >= triggerObj->size() )
      LogTrace("HLTMuonVal") <<"No HLT Collection with label "<< tag;
    else {
      size_t filterIndex = triggerObj->filterIndex(tag);
      triggerObj->getObjects( filterIndex, TriggerMuon, hltCands[i]);
    }

    hNumObjects->getTH1()->AddBinContent( i + 4, hltCands[i].size() );

  }

  for ( size_t i = 0; i < genMatches.size(); i++ ) {
    genMatches[i].l1Cand = false;
    genMatches[i].hltCands.resize(numHltLabels);
    for ( size_t j = 0; j < numHltLabels; j++ ) 
      genMatches[i].hltCands[j] = false;
  }

  for ( size_t i = 0; i < recMatches.size(); i++ ) {
    recMatches[i].l1Cand = false;
    recMatches[i].hltCands.resize(numHltLabels);
    for ( size_t j = 0; j < numHltLabels; j++ )
      recMatches[i].hltCands[j] = false;
  }

  //////////////////////////////////////////////////////////////////////////
  // Loop through L1 candidates, matching to gen/reco muons 

  unsigned int numL1Cands = 0;

  for ( size_t i = 0; i < l1Cands.size(); i++ ) {

    L1MuonParticleRef l1Cand = L1MuonParticleRef( l1Cands[i] );
    double eta   = l1Cand->eta();
    double phi   = l1Cand->phi();
    // double ptLUT = l1Cand->pt();  // L1 pt is taken from a lookup table
    // double pt    = ptLUT + 0.001; 

    double maxDeltaR = theL1DrCut;
    numL1Cands++;

    if ( m_useMuonFromGenerator ){
      int match = findGenMatch( eta, phi, maxDeltaR, genMatches );
      if ( match != -1 && genMatches[match].l1Cand == 0 ) 
	genMatches[match].l1Cand = &*l1Cand;
      else hNumOrphansGen->getTH1F()->AddBinContent( 1 );
    }

    if ( m_useMuonFromReco ){
      int match = findRecMatch( eta, phi, maxDeltaR, recMatches );
      if ( match != -1 && recMatches[match].l1Cand == 0 ) 
	recMatches[match].l1Cand = &*l1Cand;
      else hNumOrphansRec->getTH1F()->AddBinContent( 1 );
    }

  }

  if ( numL1Cands >= theNumberOfObjects ){
    if ( genMuonPt > 0 ) hPassMaxPtGen[1]->Fill(genMuonPt);
    if ( recMuonPt > 0 ) hPassMaxPtRec[1]->Fill(recMuonPt);
  }


  //////////////////////////////////////////////////////////////////////////
  // Loop through HLT candidates, matching to gen/reco muons

  for ( size_t  i = 0; i < numHltLabels; i++ ) { 
    unsigned int numFound = 0;
    for ( size_t candNum = 0; candNum < hltCands[i].size(); candNum++ ) {

      int triggerLevel = ( i < ( numHltLabels / 2 ) ) ? 2 : 3;
      double maxDeltaR = ( triggerLevel == 2 ) ? theL2DrCut : theL3DrCut;

      RecoChargedCandidateRef hltCand = hltCands[i][candNum];
      double eta = hltCand->eta();
      double phi = hltCand->phi();
      //      double pt  = hltCand->pt();

      numFound++;

      if ( m_useMuonFromGenerator ){
	int match = findGenMatch( eta, phi, maxDeltaR, genMatches );
	if ( match != -1 && genMatches[match].hltCands[i] == 0 ) 
	  genMatches[match].hltCands[i] = &*hltCand;
	else hNumOrphansGen->getTH1F()->AddBinContent( i + 2 );
      }

      if ( m_useMuonFromReco ){
	int match  = findRecMatch( eta, phi, maxDeltaR, recMatches );
	if ( match != -1 && recMatches[match].hltCands[i] == 0 ) 
	  recMatches[match].hltCands[i] = &*hltCand;
	else hNumOrphansRec->getTH1F()->AddBinContent( i + 2 );
      }

    }

    if ( numFound >= theNumberOfObjects ){
      if ( genMuonPt > 0 ) hPassMaxPtGen[i+2]->Fill( genMuonPt );
      if ( recMuonPt > 0 ) hPassMaxPtRec[i+2]->Fill( recMuonPt );
    }

  }


  //////////////////////////////////////////////////////////////////////////
  // Fill ntuple
    
  if ( m_makeNtuple ) {
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
      theNtuplePars[0] = theEventNumber;
      theNtuplePars[1] = (findMother(genMatches[i].genCand))->pdgId();
      theNtuplePars[4] = genMatches[i].genCand->pt();
      theNtuplePars[5] = genMatches[i].genCand->eta();
      theNtuplePars[6] = genMatches[i].genCand->phi();
      if ( genMatches[i].l1Cand ) {
	theNtuplePars[7] = genMatches[i].l1Cand->pt();
	theNtuplePars[8] = genMatches[i].l1Cand->eta();
	theNtuplePars[9] = genMatches[i].l1Cand->phi();
      }
      for ( size_t j = 0; j < genMatches[i].hltCands.size(); j++ ) {
	if ( genMatches[i].hltCands[j] ) {
	  if ( j == 0 ) {
	    theNtuplePars[10] = genMatches[i].hltCands[j]->pt();
	    theNtuplePars[11] = genMatches[i].hltCands[j]->eta();
	    theNtuplePars[12] = genMatches[i].hltCands[j]->phi();
	    if ( isIsolatedPath ) {
	      TrackRef tk = genMatches[i].hltCands[j]->get<TrackRef>();
	      const IsoDeposit &dep = (*caloDepMap)[tk];
	      for ( int m = 0; m < numCones; m++ ) {
		double dr = coneSizes[m];
		std::pair<double,int> depInfo = dep.depositAndCountWithin(dr);
		theNtuplePars[16+(1+numMinPtCuts)*2*m+0] = depInfo.first;
		theNtuplePars[16+(1+numMinPtCuts)*2*m+1] = depInfo.second;
	  } } }
	  if ( ( !isIsolatedPath && j == 1 ) ||
	       (  isIsolatedPath && j == 2 ) ) {
	    theNtuplePars[13] = genMatches[i].hltCands[j]->pt();
	    theNtuplePars[14] = genMatches[i].hltCands[j]->eta();
	    theNtuplePars[15] = genMatches[i].hltCands[j]->phi();
	    if ( isIsolatedPath ) {
	      TrackRef tk = genMatches[i].hltCands[j]->get<TrackRef>();
	      const IsoDeposit &dep = (*trackDepMap)[tk];
	      for ( int m = 0; m < numCones; m++ ) {
		for ( int n = 0; n < numMinPtCuts; n++ ) {
		  double dr = coneSizes[m];
		  double minPt = minPtCuts[n];
		  std::pair<double,int> depInfo;
		  depInfo = dep.depositAndCountWithin(dr, vetos, minPt);
		  theNtuplePars[16+(1+numMinPtCuts)*2*m+2+2*n] =depInfo.first;
		  theNtuplePars[16+(1+numMinPtCuts)*2*m+3+2*n] =depInfo.second;
		}
	  } } }
	  if ( isIsolatedPath && j == 1 ) theNtuplePars[2] = true;
	  if ( isIsolatedPath && j == 3 ) theNtuplePars[3] = true;
	}
      }
      theNtuple->Fill(theNtuplePars); 
    }
  }
  
  //////////////////////////////////////////////////////////////////////////
  // Fill histograms
    
  for ( size_t i = 0; i < genMatches.size(); i++ ) {
    double pt  = genMatches[i].genCand->pt();
    double eta = genMatches[i].genCand->eta();
    double phi = genMatches[i].genCand->phi();
    if ( pt > theMinPtCut &&  fabs(eta) < theMaxEtaCut ) {
      hNumObjects->getTH1()->AddBinContent(1);
      hPassPtGen[0]->Fill(pt);
      hPassEtaGen[0]->Fill(eta);
      hPassPhiGen[0]->Fill(phi);
      if ( genMatches[i].l1Cand ) {
	hPassPtGen[1]->Fill(pt);
	hPassEtaGen[1]->Fill(eta);
	hPassPhiGen[1]->Fill(phi);
	for ( size_t j = 0; j < genMatches[i].hltCands.size(); j++ ) {
	  bool foundAllPreviousCands = true;
	  for ( size_t k = 0; k < j; k++ ) 
	    if ( !genMatches[i].hltCands[k] ) 
	      foundAllPreviousCands = false;
	  if ( foundAllPreviousCands && genMatches[i].hltCands[j] ) {
	    hPassPtGen[j+2]->Fill(pt);
	    hPassEtaGen[j+2]->Fill(eta);
	    hPassPhiGen[j+2]->Fill(phi);
  } } } } }

  for ( size_t i = 0; i < recMatches.size(); i++ ) {
    double pt  = recMatches[i].recCand->pt();
    double eta = recMatches[i].recCand->eta();
    double phi = recMatches[i].recCand->phi();
    if ( pt > theMinPtCut &&  fabs(eta) < theMaxEtaCut ) {
      hNumObjects->getTH1()->AddBinContent(2);
      hPassPtRec[0]->Fill(pt);
      hPassEtaRec[0]->Fill(eta);
      hPassPhiRec[0]->Fill(phi);
      if ( recMatches[i].l1Cand ) {
	hPassPtRec[1]->Fill(pt);
	hPassEtaRec[1]->Fill(eta);
	hPassPhiRec[1]->Fill(phi);
	for ( size_t j = 0; j < recMatches[i].hltCands.size(); j++ ) {
	  bool foundAllPreviousCands = true;
	  for ( size_t k = 0; k < j; k++ ) 
	    if ( !recMatches[i].hltCands[k] ) 
	      foundAllPreviousCands = false;
	  if ( foundAllPreviousCands && recMatches[i].hltCands[j] ) {
	    hPassPtRec[j+2]->Fill(pt);
	    hPassEtaRec[j+2]->Fill(eta);
	    hPassPhiRec[j+2]->Fill(phi);
  } } } } }

}



const reco::Candidate* 
HLTMuonGenericRate::findMother( const reco::Candidate* p ) 
{
  const reco::Candidate* mother = p->mother();
  if ( mother ) {
    if ( mother->pdgId() == p->pdgId() ) return findMother(mother);
    else return mother;
  }
  else return 0;
}



int 
HLTMuonGenericRate::findGenMatch( double eta, double phi, double maxDeltaR,
				  vector<MatchStruct> matches )
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



int 
HLTMuonGenericRate::findRecMatch( double eta, double phi,  double maxDeltaR,
				  vector<MatchStruct> matches )
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



void 
HLTMuonGenericRate::begin() 
{

  TString myLabel, newFolder;
  vector<TH1F*> h;

  if (dbe_) {
    dbe_->cd();
    dbe_->setCurrentFolder("HLT/Muon");

    myLabel = theL1CollectionLabel;
    myLabel = myLabel(myLabel.Index("L1"),myLabel.Length());
    myLabel = myLabel(0,myLabel.Index("Filtered")+8);

    newFolder = "HLT/Muon/Distributions/" + theTriggerName;
    dbe_->setCurrentFolder( newFolder.Data() );

    NumberOfEvents     = dbe_->bookInt("NumberOfEvents");
    MinPtCut           = dbe_->bookFloat("MinPtCut");
    MaxEtaCut          = dbe_->bookFloat("MaxEtaCut");
    MinPtCut ->Fill(theMinPtCut );
    MaxEtaCut->Fill(theMaxEtaCut);
    
    vector<string> binLabels;
    binLabels.push_back( theL1CollectionLabel.c_str() );
    for ( size_t i = 0; i < theHltCollectionLabels.size(); i++ )
      binLabels.push_back( theHltCollectionLabels[i].c_str() );

    hNumObjects = dbe_->book1D( "numObjects", "Number of Objects", 7, 0, 7 );
    hNumObjects->setBinLabel( 1, "Gen" );
    hNumObjects->setBinLabel( 2, "Reco" );
    for ( size_t i = 0; i < binLabels.size(); i++ )
      hNumObjects->setBinLabel( i + 3, binLabels[i].c_str() );

    if (m_useMuonFromGenerator){

      hNumOrphansGen = dbe_->book1D( "genNumOrphans", "Number of Orphans;;Number of objects not matched to a gen muon", 5, 0, 5 );
      for ( size_t i = 0; i < binLabels.size(); i++ )
	hNumOrphansGen->setBinLabel( i + 1, binLabels[i].c_str() );

      hPassMaxPtGen.push_back( bookIt( "genPassMaxPt_All", "Highest Gen Muon Pt", theMaxPtParameters) );
      hPassMaxPtGen.push_back( bookIt( "genPassMaxPt_" + myLabel, "Highest Gen Muon pt >= 1 L1 Candidate, label=" + myLabel, theMaxPtParameters) );
      hPassPtGen.push_back( bookIt( "genPassPt_All", "Gen Muon Pt",  thePtParameters) );
      hPassPtGen.push_back( bookIt( "genPassPt_" + myLabel, "Gen Muon pt >= 1 L1 Candidate, label=" + myLabel,  thePtParameters) );
      hPassEtaGen.push_back( bookIt( "genPassEta_All", "#eta of Gen Muons", theEtaParameters) );
      hPassEtaGen.push_back( bookIt( "genPassEta_" + myLabel, "#eta of Gen Muons Matched to L1, label=" + myLabel, theEtaParameters) );
      hPassPhiGen.push_back( bookIt( "genPassPhi_All", "#phi of Gen Muons", thePhiParameters) );
      hPassPhiGen.push_back( bookIt( "genPassPhi_" + myLabel, "#phi of Gen Muons Matched to L1, label=" + myLabel, thePhiParameters) );

    }

    if (m_useMuonFromReco){

      hNumOrphansRec = dbe_->book1D( "recNumOrphans", "Number of Orphans;;Number of objects not matched to a reconstructed muon", 5, 0, 5 );
      for ( size_t i = 0; i < binLabels.size(); i++ )
	hNumOrphansRec->setBinLabel( i + 1, binLabels[i].c_str() );

      hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_All", "Highest Reco Muon Pt" + myLabel,  theMaxPtParameters) );
      hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_" + myLabel, "pt of Reco Muons Matched to L1, label=" + myLabel,  theMaxPtParameters) );
      hPassPtRec.push_back( bookIt( "recPassPt_All", "pt of Reco Muons" + myLabel,  thePtParameters) );
      hPassPtRec.push_back( bookIt( "recPassPt_" + myLabel, "pt of Reco Muons Matched to L1, label=" + myLabel,  thePtParameters) );
      hPassEtaRec.push_back( bookIt( "recPassEta_All", "#eta of Reco Muons", theEtaParameters) );
      hPassEtaRec.push_back( bookIt( "recPassEta_" + myLabel, "#eta of Reco Muons Matched to L1, label=" + myLabel, theEtaParameters) );
      hPassPhiRec.push_back( bookIt( "recPassPhi_All", "#phi of Reco Muons", thePhiParameters) );
      hPassPhiRec.push_back( bookIt( "recPassPhi_" + myLabel, "#phi of Reco Muons Matched to L1, label=" + myLabel, thePhiParameters) );

    }

    for (unsigned int i = 0; i < theHltCollectionLabels.size(); i++) {

      myLabel = theHltCollectionLabels[i];
      TString level = ( myLabel.Contains("L2") ) ? "L2" : "L3";
      myLabel = myLabel(myLabel.Index(level),myLabel.Length());
      myLabel = myLabel(0,myLabel.Index("Filtered")+8);
      
      if (m_useMuonFromGenerator) {

	hPassMaxPtGen.push_back( bookIt( "genPassMaxPt_" + myLabel, "Highest Gen Muon pt with >= 1 Candidate, label=" + myLabel, theMaxPtParameters) );   
	hPassPtGen.push_back( bookIt( "genPassPt_" + myLabel, "Highest Gen Muon pt with >= 1 Candidate, label=" + myLabel, thePtParameters) );   
	hPassEtaGen.push_back( bookIt( "genPassEta_" + myLabel, "#eta of Gen Muons Matched to HLT, label=" + myLabel, theEtaParameters) );
	hPassPhiGen.push_back( bookIt( "genPassPhi_" + myLabel, "#phi of Gen Muons Matched to HLT, label=" + myLabel, thePhiParameters) );

      }

      if (m_useMuonFromReco) {

	hPassMaxPtRec.push_back( bookIt( "recPassMaxPt_" + myLabel, "pt of Reco Muons Matched to HLT, label=" + myLabel, theMaxPtParameters) );     
	hPassPtRec.push_back( bookIt( "recPassPt_" + myLabel, "pt of Reco Muons Matched to HLT, label=" + myLabel, thePtParameters) );     
	hPassEtaRec.push_back( bookIt( "recPassEta_" + myLabel, "#eta of Reco Muons Matched to HLT, label=" + myLabel, theEtaParameters) );
	hPassPhiRec.push_back( bookIt( "recPassPhi_" + myLabel, "#phi of Reco Muons Matched to HLT, label=" + myLabel, thePhiParameters) );
      }

    }
  }

}



MonitorElement* 
HLTMuonGenericRate::bookIt(TString name, TString title, vector<double> params)
{
  LogDebug("HLTMuonVal") << "Directory " << dbe_->pwd() << " Name " << 
                            name << " Title:" << title;
  int nBins  = (int)params[0];
  double min = params[1];
  double max = params[2];
  TH1F *h = new TH1F( name, title, nBins, min, max );
  h->Sumw2();
  return dbe_->book1D( name.Data(), h );
  delete h;
}

