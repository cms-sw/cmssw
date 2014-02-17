// -*- C++ -*-
//
// Package:    L25TauAnalyzer
// Class:      L25TauAnalyzer
// 
/**\class L25TauAnalyzer L25TauAnalyzer.cc HLTriggerOffline/L25TauAnalyzer/src/L25TauAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Eduardo Luiggi
//         Created:  Mon Apr 19 16:09:13 CDT 2010
// $Id: L25TauAnalyzer.cc,v 1.11 2011/03/01 22:54:26 eluiggi Exp $
//
//


#include "HLTriggerOffline/Tau/interface/L25TauAnalyzer.h"
#include "Math/GenVector/VectorUtil.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace reco;
using namespace std;

L25TauAnalyzer::L25TauAnalyzer(const edm::ParameterSet& iConfig){
   //now do what ever initialization is needed

  _l25JetSource = iConfig.getParameter<InputTag>("L25JetSource");
  _l2TauInfoAssoc = iConfig.getParameter<InputTag>("L2TauSource");
  _pfTauSource = iConfig.getParameter<InputTag>("PFTauSource");
  _pfTauIsoSource = iConfig.getParameter<InputTag>("PFTauIsoSource");
  _pfTauMuonDiscSource = iConfig.getParameter<InputTag>("PFTauMuonDiscSource");
  _pVtxSource = iConfig.getParameter<InputTag>("PrimaryVtxSource");
  _l2l25MatchingCone = iConfig.getParameter<double>("L2L25MatchingCone");
  
  _l25JetLeadTkMacthingCone =  iConfig.getParameter<double>("L25JetLeadTkMatchingCone");
  _minTrackPt = iConfig.getParameter<double>("MinTrackPt");
  _signalCone = iConfig.getParameter<double>("SignalCone");
  _isolationCone = iConfig.getParameter<double>("IsolationCone");
  _l25Dz = iConfig.getParameter<double>("L25DzCut");
  _nTrkIso = iConfig.getParameter<int>("NTrkIso");
  _l25LeadTkPtMin = iConfig.getParameter<double>("L25LeadTkMinPt");
  
     
  l25JetEt=0.;
  l25JetEta=10.;
  l25JetPhi=10.;
  hasLeadTrk=0;
  leadSignalTrackPt = 0.;
  leadTrkJetDeltaR = 0.;
  numPixTrkInJet = 0;
  numQPixTrkInJet = 0;
  numQPixTrkInSignalCone = 0;
  numQPixTrkInAnnulus = 10;
  
  pfTauPt = 0.;
  pfTauEt = 0.;
  pfTauEta = 10.;
  pfTauPhi = 10.;
  pfTauLTPt = 0.;
  pfTauTrkIso = 100.;
  pfTauGammaIso = 100.;
  
  l2JetEt = 0;
  l2JetEta = 10;
  l2JetPhi = 10;
  
  l25MatchedToPFTau = 0;
  L2MatchedToPFtau = 0;
  L25MatchedToL2 = 0;
  
  initializeVectors();
  edm::Service<TFileService> fs;
  l25tree = fs->make<TTree>("l25tree","Level 2.5 Tau Tree");
  
  l25tree->Branch("pfTauHasLeadTrk", &pfTauHasLeadTrk, "pfTauHasLeadTrk/B");
  l25tree->Branch("pfTauInBounds", &pfTauInBounds, "pfTauInBounds/B");
  l25tree->Branch("pfTauPt", &pfTauPt, "pfTauPt/F");
  l25tree->Branch("pfTauEt", &pfTauEt, "pfTauEt/F");
  l25tree->Branch("pfTauEta", &pfTauEta, "pfTauEta/F");
  l25tree->Branch("pfTauphi", &pfTauPhi, "pfTauPhi/F");
  l25tree->Branch("pfTauLTPt", &pfTauLTPt, "pfTauLTPt/F");
  l25tree->Branch("pfTauIsoDisc", &pfTauIsoDisc, "pfTauIsoDisc/F");
  l25tree->Branch("pfTauMuonDisc", &pfTauMuonDisc, "pfTauMuonDisc/F");
  l25tree->Branch("pfTauNProngs", &pfTauNProngs, "pfTauNProngs/I");
  l25tree->Branch("pfTauTrkIso", &pfTauTrkIso, "pfTauTrkIso/F");
  l25tree->Branch("pfTauNTrkIso", &pfTauNTrkIso, "pfTauNTrkIso/I");
  l25tree->Branch("pfTauIsoTrkPt", &pfTauIsoTrkPt, "pfTauIsoTrkPt/F");
  l25tree->Branch("pfTauGammaIso", &pfTauGammaIso, "pfTauGammaIso/F");
  
  l25tree->Branch("pftauL25DeltaR", &pftauL25DeltaR, "pftauL25DeltaR/F");
  l25tree->Branch("pftauSignalTrkDeltaR", &pftauSignalTrkDeltaR, "pftauSignalTrkDeltaR/F");
  l25tree->Branch("pftauIsoTrkDeltaR", &pftauIsoTrkDeltaR, "pftauIsoTrkDeltaR/F");
  l25tree->Branch("leadTrkPtRes", &leadTrkPtRes, "leadTrkPtRes/F");
  l25tree->Branch("leadTrkDeltaR", &leadTrkDeltaR, "leadTrkDeltaR/F");
  l25tree->Branch("leadIsoTrkPtRes", &leadIsoTrkPtRes, "leadIsoTrkPtRes/F");
  l25tree->Branch("leadIsoTrkDeltaR", &leadIsoTrkDeltaR, "leadIsoTrkDeltaR/F");
  l25tree->Branch("l25Disc_LeadTkDir", &l25Disc_LeadTkDir, "l25Disc_LeadTkDir/B");
  l25tree->Branch("l25Disc_JetDir", &l25Disc_JetDir, "l25Disc_JetDir/B");
 
  l25tree->Branch("l2JetEt", &l2JetEt, "l2JetEt/F");
  l25tree->Branch("l2JetEta", &l2JetEta, "l2JetEta/F");
  l25tree->Branch("l2JetPhi", &l2JetPhi, "l2JetPhi/F");
 
  l25tree->Branch("l25MatchedToPFTau", &l25MatchedToPFTau, "l25MatchedToPFTau/B");
  l25tree->Branch("L2MatchedToPFtau", &L2MatchedToPFtau, "L2MatchedToPFtau/B");
  l25tree->Branch("L25MatchedToL2", &L25MatchedToL2, "L25MatchedToL2/B");
  
  l25tree->Branch("l25JetEt", &l25JetEt, "l25JetEt/F");
  l25tree->Branch("l25JetEta", &l25JetEta, "l25JetEta/F");
  l25tree->Branch("l25JetPhi", &l25JetPhi, "l25JetPhi/F");
  l25tree->Branch("hasLeadTrack", &hasLeadTrk, "hasLeadTrack/B");
  l25tree->Branch("leadTrackPt", &leadSignalTrackPt, "leadTrackPt/F");
  l25tree->Branch("nTracks", &numPixTrkInJet, "nTracks/I");
  l25tree->Branch("nQTracks", &numQPixTrkInJet, "nQTracks/I");
  l25tree->Branch("nQTracksInSignal", &numQPixTrkInSignalCone, "nQTracksInSignal/I");
  l25tree->Branch("nQTracksInAnnulus", &numQPixTrkInAnnulus, "nQTracksInAnnulus/I");
  
  l25tree->Branch("l25TrkPt", &l25TrkPt);
  l25tree->Branch("l25TrkEta", &l25TrkEta);
  l25tree->Branch("l25TrkPhi", &l25TrkPhi);
  l25tree->Branch("l25TrkDz", &l25TrkDz);
  l25tree->Branch("l25TrkDxy", &l25TrkDxy);
  l25tree->Branch("l25TrkChi2", &l25TrkChi2);
  l25tree->Branch("l25TrkChi2NdF", &l25TrkChi2NdF);
  l25tree->Branch("l25TrkNRecHits", &l25TrkNRecHits);
  l25tree->Branch("l25TrkNValidPixelHits", &l25TrkNValidPixelHits);
  
  l25tree->Branch("l25SignalTrkPt", &l25SignalTrkPt);
  l25tree->Branch("l25SignalTrkEta", &l25SignalTrkEta);
  l25tree->Branch("l25SignalTrkPhi", &l25SignalTrkPhi);
  l25tree->Branch("l25SignalTrkDz", &l25SignalTrkDz);
  l25tree->Branch("l25SignalTrkDxy", &l25SignalTrkDxy);
  l25tree->Branch("l25SignalTrkChi2", &l25SignalTrkChi2);
  l25tree->Branch("l25SignalTrkChi2NdF", &l25SignalTrkChi2NdF);
  l25tree->Branch("l25SignalTrkNRecHits", &l25SignalTrkNRecHits);
  l25tree->Branch("l25SignalTrkNValidHits", &l25SignalTrkNValidHits);
  l25tree->Branch("l25SignalTrkNValidPixelHits", &l25SignalTrkNValidPixelHits);
  l25tree->Branch("l25SignalTrkNLostHits", &l25SignalTrkNLostHits);
  
  l25tree->Branch("l25IsoTrkPt", &l25IsoTrkPt);
  l25tree->Branch("l25IsoTrkEta", &l25IsoTrkEta);
  l25tree->Branch("l25IsoTrkPhi", &l25IsoTrkPhi);
  l25tree->Branch("l25IsoTrkDz", &l25IsoTrkDz);
  l25tree->Branch("l25IsoTrkDxy", &l25IsoTrkDxy);
  l25tree->Branch("l25IsoTrkChi2", &l25IsoTrkChi2);
  l25tree->Branch("l25IsoTrkChi2NdF", &l25IsoTrkChi2NdF);
  l25tree->Branch("l25IsoTrkNRecHits", &l25IsoTrkNRecHits);
  l25tree->Branch("l25IsoTrkNValidHits", &l25IsoTrkNValidHits);
  l25tree->Branch("l25IsoTrkNValidPixelHits", &l25IsoTrkNValidPixelHits);
  l25tree->Branch("l25IsoTrkNLostHits", &l25IsoTrkNLostHits);
  
  l25tree->Branch("myNtrkIso", &myNtrkIso, "myNtrkIso/I");
}


L25TauAnalyzer::~L25TauAnalyzer(){
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void L25TauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
   using namespace edm;

  //get l2 jet; match to pftau
  Handle<PFTauCollection> thePFTaus;
  iEvent.getByLabel(_pfTauSource,thePFTaus);
  
  Handle<PFTauDiscriminator> thePFTauDiscriminatorByIsolation;
  iEvent.getByLabel(_pfTauIsoSource,thePFTauDiscriminatorByIsolation);  
  
  Handle<PFTauDiscriminator> thePFTauDiscriminatorAgainstMuon;
  iEvent.getByLabel(_pfTauMuonDiscSource, thePFTauDiscriminatorAgainstMuon); 
  
  Handle<VertexCollection> thePrimaryVertices;
  iEvent.getByLabel(_pVtxSource, thePrimaryVertices);
  const reco::Vertex& theHltPrimaryVertex = (*thePrimaryVertices->begin());
  theVertexPosition = math::XYZPoint(0.,0.,0.);
  if(thePrimaryVertices->size() > 0) theVertexPosition = math::XYZPoint(theHltPrimaryVertex.position().x(), 
                                                     theHltPrimaryVertex.position().y(),
                                                     theHltPrimaryVertex.position().z());  
  
  //Loop over PFtaus
  if(thePFTaus.isValid()){
    edm::LogInfo("L25Analysis")  << "Inside PFTau\n";
    for(unsigned int pfTauIt = 0; pfTauIt < thePFTaus->size(); ++pfTauIt){
      const PFTauRef thisTauRef(thePFTaus,pfTauIt);
      pfTauIsoDisc = 0;
      pfTauMuonDisc = 0;
      if(thePFTauDiscriminatorByIsolation.isValid()){
        const PFTauDiscriminator& disc = *(thePFTauDiscriminatorByIsolation.product());
	pfTauIsoDisc = disc[thisTauRef];
      }
      if(thePFTauDiscriminatorAgainstMuon.isValid()){
        const PFTauDiscriminator& disc = *(thePFTauDiscriminatorAgainstMuon.product());
	pfTauMuonDisc = disc[thisTauRef];
      }
      
      L2MatchedToPFtau = false;      
      pfTauHasLeadTrk = false;
      pfTauInBounds = true;
      
      pfTauPt = thePFTaus->at(pfTauIt).pt();							  
      pfTauEt = thePFTaus->at(pfTauIt).et();							  
      pfTauEta = thePFTaus->at(pfTauIt).eta(); 						  
      pfTauPhi = thePFTaus->at(pfTauIt).eta(); 
      
      const PFCandidateRef& thePFTauLeadTrack = thePFTaus->at(pfTauIt).leadPFChargedHadrCand(); 
      if(thePFTauLeadTrack.isNonnull()){
        pfTauHasLeadTrk = true;
        pfTauNProngs = thePFTaus->at(pfTauIt).signalPFChargedHadrCands().size();			      
        pfTauLTPt = thePFTaus->at(pfTauIt).leadPFChargedHadrCand()->pt();
        pfTauTrkIso = thePFTaus->at(pfTauIt).isolationPFChargedHadrCandsPtSum();
        pfTauGammaIso = thePFTaus->at(pfTauIt).isolationPFGammaCandsEtSum();
      }
      
      const PFCandidateRefVector& theSignalCands = thePFTaus->at(pfTauIt).signalPFChargedHadrCands();
      for(PFCandidateRefVector::const_iterator vIt = theSignalCands.begin(); vIt != theSignalCands.end(); ++vIt){
        pftauSignalTrkDeltaR = ROOT::Math::VectorUtil::DeltaR((*vIt)->trackRef()->momentum(), thePFTauLeadTrack->momentum());
	if(pftauSignalTrkDeltaR > _signalCone) pfTauInBounds = false;
      }
    
      const PFCandidateRefVector& theIsoCands = thePFTaus->at(pfTauIt).isolationPFChargedHadrCands();
      pfTauNTrkIso = theIsoCands.size();
      float PFTauLeadIsoPt = 0.;
      Track thePFTauLeadIsoTrk;      
      for(PFCandidateRefVector::const_iterator vIt = theIsoCands.begin(); vIt != theIsoCands.end(); ++vIt){
        pftauIsoTrkDeltaR = ROOT::Math::VectorUtil::DeltaR((*vIt)->trackRef()->momentum(), thePFTauLeadTrack->momentum());
	if(pftauIsoTrkDeltaR < _isolationCone) pfTauInBounds = false;
	pfTauIsoTrkPt = (*vIt)->trackRef()->pt();		      
	if((*vIt)->trackRef()->pt() > PFTauLeadIsoPt){
	  thePFTauLeadIsoTrk = *((*vIt)->trackRef());
	  PFTauLeadIsoPt = (*vIt)->trackRef()->pt();
	}
      }
      
      // matched PFtau to l2 Jet with Et > 5 ... originially 15 but to strong for min bias
      Handle<L2TauInfoAssociation> l2TauInfoAssoc; //Handle to the input (L2 Tau Info Association) 
      iEvent.getByLabel(_l2TauInfoAssoc,l2TauInfoAssoc);					   
      reco::CaloJet theMatchedL2Jet;
      IsolatedTauTagInfo theMatchedL25TauTagInfo;
      if(iEvent.getByLabel(_l2TauInfoAssoc,l2TauInfoAssoc) && l2TauInfoAssoc->size()>0){
      	double matchDr = _l2l25MatchingCone;
      	for(L2TauInfoAssociation::const_iterator it = l2TauInfoAssoc->begin(); it != l2TauInfoAssoc->end(); ++it){
      	  const reco::CaloJet& l2Jet =*(it->key);
      	  if(l2Jet.et() < 15.) continue;
      	  double delta = ROOT::Math::VectorUtil::DeltaR(thePFTaus->at(pfTauIt).p4(),l2Jet.p4());
      	  if(delta < matchDr){
      	    matchDr = delta;
	    L2MatchedToPFtau = true;
      	    theMatchedL2Jet = l2Jet;
      	  }
      	}
      }
      
      //if(L2MatchedToPFtau) match to l25 and fill l25 variables
      Handle<IsolatedTauTagInfoCollection> tauTagInfo;
      if(L2MatchedToPFtau){
        l2JetEt = theMatchedL2Jet.et();
        l2JetEta = theMatchedL2Jet.eta();
        l2JetPhi = theMatchedL2Jet.phi();
        if(iEvent.getByLabel(_l25JetSource, tauTagInfo) && tauTagInfo->size()>0){
	  L25MatchedToL2 = false;									       
	  double minDeltaR = _l2l25MatchingCone;								       
	  //declare l25 tauTagInfo ...									       
	  //IsolatedTauTagInfo theMatchedL25TauTagInfo;							       
	  for(IsolatedTauTagInfoCollection::const_iterator i = tauTagInfo->begin();i!=tauTagInfo->end();++i){ 
	    double delta = ROOT::Math::VectorUtil::DeltaR(theMatchedL2Jet.p4(), i->jet()->p4());	       
	    if(delta < minDeltaR){									       
	      minDeltaR = delta;									       
	      L25MatchedToL2 = true;									       
	      theMatchedL25TauTagInfo = *i;
	    }												       
	  }												       
	}
	
	pftauL25DeltaR = ROOT::Math::VectorUtil::DeltaR(thePFTaus->at(pfTauIt).p4(), theMatchedL25TauTagInfo.jet()->p4());
	if(pftauL25DeltaR < _l2l25MatchingCone)l25MatchedToPFTau = true;
	
 	const TrackRefVector theTauTagTracks = theMatchedL25TauTagInfo.allTracks();
	
   	l25JetEt = theMatchedL25TauTagInfo.jet()->et();   													     
     	l25JetEta = theMatchedL25TauTagInfo.jet()->eta();
     	l25JetPhi = theMatchedL25TauTagInfo.jet()->phi();
	numPixTrkInJet = theTauTagTracks.size();										    	     
     	numQPixTrkInJet = theMatchedL25TauTagInfo.selectedTracks().size();
	//get info for all tracks in tauTagInfo
	for(TrackRefVector::const_iterator trkIt = theTauTagTracks.begin(); trkIt != theTauTagTracks.end(); ++trkIt){
	  l25TrkPt->push_back((*trkIt)->pt());
      	  l25TrkEta->push_back((*trkIt)->eta());
      	  l25TrkPhi->push_back((*trkIt)->phi());
      	  l25TrkDz->push_back((*trkIt)->dz(theVertexPosition));
      	  l25TrkDxy->push_back((*trkIt)->dxy(theVertexPosition));
	  l25TrkChi2->push_back((*trkIt)->chi2());
	  l25TrkChi2NdF->push_back((*trkIt)->normalizedChi2());
	  l25TrkNRecHits->push_back((*trkIt)->recHitsSize());
	  l25TrkNValidPixelHits->push_back((*trkIt)->hitPattern().numberOfValidPixelHits());
	}
	const TrackRef leadTk = theMatchedL25TauTagInfo.leadingSignalTrack(_l25JetLeadTkMacthingCone, _minTrackPt);									 
      	if(!leadTk){															 	    
     	  hasLeadTrk=false;	   													    
     	  numQPixTrkInSignalCone=0;													    
     	  numQPixTrkInAnnulus=0;   													    
     	  leadSignalTrackPt=0;     													    
     	  leadTrkJetDeltaR=10;
	  leadTrkDeltaR=10;   													    
     	}															 	    
     	else{															 	    
     	  hasLeadTrk=true;														  
     	  leadTrkJetDeltaR = ROOT::Math::VectorUtil::DeltaR(theMatchedL25TauTagInfo.jet()->p4().Vect(), leadTk->momentum());
     	  leadSignalTrackPt=leadTk->pt();
	  if(pfTauLTPt != 0)leadTrkPtRes = (leadTk->pt() - pfTauLTPt) / pfTauLTPt;
	  leadTrkDeltaR = ROOT::Math::VectorUtil::DeltaR(thePFTauLeadTrack->momentum(), leadTk->momentum());
	  
	  //print info in the case where the pftau is isolated but the tauTag is not
	  bool l25IsoDisc = theMatchedL25TauTagInfo.discriminator(0.2,0.15,0.5,5.,1.,0,0.2);
	  if(pfTauTrkIso <= 1. && l25MatchedToPFTau && !l25IsoDisc) printInfo(thePFTaus->at(pfTauIt), theMatchedL25TauTagInfo);
	  
	  const TrackRefVector theSignalTracks = theMatchedL25TauTagInfo.tracksInCone(leadTk->momentum(), _signalCone, _minTrackPt);
	  const TrackRefVector theIsoTracks = theMatchedL25TauTagInfo.tracksInCone(leadTk->momentum(), _isolationCone, _minTrackPt);
	  numQPixTrkInSignalCone = theSignalTracks.size();				  
     	  if(numQPixTrkInSignalCone > 0) numQPixTrkInAnnulus = theIsoTracks.size() - theSignalTracks.size();
	  //get the lead track in isolation 
	  float l25LeadIsoPt = 0.;
	  Track theL25LeadIsoTrk;
	  
	  myNtrkIso = 0;
	  for(TrackRefVector::const_iterator isoIt = theIsoTracks.begin(); isoIt != theIsoTracks.end(); ++isoIt){
	    if(ROOT::Math::VectorUtil::DeltaR(leadTk->momentum(), (*isoIt)->momentum()) <= _signalCone){
	  
	      l25SignalTrkPt->push_back((*isoIt)->pt());
	      l25SignalTrkEta->push_back((*isoIt)->eta());
	      l25SignalTrkPhi->push_back((*isoIt)->phi());
	      l25SignalTrkDz->push_back((*isoIt)->dz(theVertexPosition));
	      l25SignalTrkDxy->push_back((*isoIt)->dxy(theVertexPosition));
	      l25SignalTrkChi2->push_back((*isoIt)->chi2());
	      l25SignalTrkChi2NdF->push_back((*isoIt)->normalizedChi2());
	      l25SignalTrkNRecHits->push_back((*isoIt)->recHitsSize());
	      l25SignalTrkNValidHits->push_back((*isoIt)->numberOfValidHits());
	      l25SignalTrkNValidPixelHits->push_back((*isoIt)->hitPattern().numberOfValidPixelHits());
	      l25SignalTrkNLostHits->push_back((*isoIt)->lost());

	    }
	    else{
	      myNtrkIso++;

	      l25IsoTrkPt->push_back((*isoIt)->pt());
	      l25IsoTrkChi2NdF->push_back((*isoIt)->normalizedChi2());
	      l25IsoTrkChi2->push_back((*isoIt)->chi2());
	      l25IsoTrkNValidHits->push_back((*isoIt)->numberOfValidHits());
	      l25IsoTrkNRecHits->push_back((*isoIt)->recHitsSize());
	      l25IsoTrkNValidPixelHits->push_back((*isoIt)->hitPattern().numberOfValidPixelHits());
	      l25IsoTrkNLostHits->push_back((*isoIt)->lost());
	      l25IsoTrkDxy->push_back((*isoIt)->dxy(theVertexPosition));
	      l25IsoTrkDz->push_back((*isoIt)->dz(theVertexPosition));
	      l25IsoTrkEta->push_back((*isoIt)->eta());
	      l25IsoTrkPhi->push_back((*isoIt)->phi());
	      if((*isoIt)->pt() > l25LeadIsoPt){
	        theL25LeadIsoTrk = **isoIt;
	        l25LeadIsoPt = (*isoIt)->pt();
	      }
	    }
	  }
	  //if((nIsoTraks > numQPixTrkInAnnulus) ? std::cout << "FUCK \n" : std::cout << "GREAT\n" );
	  if(thePFTauLeadIsoTrk.pt() != 0) leadIsoTrkPtRes = (theL25LeadIsoTrk.pt() - thePFTauLeadIsoTrk.pt()) /thePFTauLeadIsoTrk.pt();
	  leadIsoTrkDeltaR = ROOT::Math::VectorUtil::DeltaR(theL25LeadIsoTrk.momentum(), thePFTauLeadIsoTrk.momentum());
	  l25Disc_LeadTkDir = theMatchedL25TauTagInfo.discriminator(leadTk->momentum(), _l25JetLeadTkMacthingCone, _signalCone, _isolationCone, _l25LeadTkPtMin, _minTrackPt, _nTrkIso, _l25Dz);
	  l25Disc_JetDir = theMatchedL25TauTagInfo.discriminator(theMatchedL25TauTagInfo.jet()->momentum(), _l25JetLeadTkMacthingCone, _signalCone, _isolationCone, _l25LeadTkPtMin, _minTrackPt, _nTrkIso, _l25Dz);
     	}
      }
      //Fill here per pfTau ...
     l25tree->Fill();
     clearVectors();
    }
  }
  else std::cout << "Invalid PFTau Collection\n"; 
}



reco::PFTau L25TauAnalyzer::match(const reco::Jet& jet, const reco::PFTauCollection& thePFTauColl){

  //Loop On the Collection and see if your tau jet is matched to one there
  //Also find the nearest Matched MC Particle to your Jet (to be complete)
  // switch reference collection to pftau;  match to this.
 
 reco::PFTau theMatchedPFTau;
 double matchDr=0.3;
 
 if(thePFTauColl.size()>0)
  for(reco::PFTauCollection::const_iterator it = thePFTauColl.begin(); it != thePFTauColl.end(); ++it){
    double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4(),(*it).p4()); 	 
    if(delta<matchDr){
      matchDr = delta;
      theMatchedPFTau = *it;							      
    } 									 
  }
  return theMatchedPFTau;
}

reco::CaloJet L25TauAnalyzer::matchedToPFTau(const reco::PFTau& thePFTau, const L2TauInfoAssociation& theL2Info){
  double matchDr = 0.5;
  reco::CaloJet theMatchedJet;
  for(L2TauInfoAssociation::const_iterator it = theL2Info.begin(); it != theL2Info.end(); ++it){
    const reco::CaloJet& l2Jet =*(it->key);
    if(l2Jet.et() < 15.) continue;
    double delta = ROOT::Math::VectorUtil::DeltaR(thePFTau.p4(),l2Jet.p4());
    if(delta < matchDr){
      matchDr = delta;
      theMatchedJet = l2Jet;
    }
  }
  return theMatchedJet;
}

void L25TauAnalyzer::printInfo(const reco::PFTau& thePFTau, const reco::IsolatedTauTagInfo& theTauTagInfo){
  const TrackRef theLeadTrack = theTauTagInfo.leadingSignalTrack(_l25JetLeadTkMacthingCone, _minTrackPt);
  const TrackRefVector theSignalTracks = theTauTagInfo.tracksInCone(theLeadTrack->momentum(), _signalCone, _minTrackPt);
  const TrackRefVector theIsoTracks = theTauTagInfo.tracksInCone(theLeadTrack->momentum(), _isolationCone, _minTrackPt);
  
  std::cout << " Isolated PFTau Matched to Non-Isolated L25 Object( PFTau:L25)"
  << "\n Et\t" << thePFTau.et() << "\t" << theTauTagInfo.jet()->et()
  << "\n Eta\t" << thePFTau.eta() << "\t" << theTauTagInfo.jet()->eta()
  << "\n Phi\t" << thePFTau.phi() << "\t" << theTauTagInfo.jet()->phi()
  << "\n LTPt\t" << thePFTau.leadPFChargedHadrCand()->pt() << "\t" << theLeadTrack->pt()
  << "\n LTEta\t" << thePFTau.leadPFChargedHadrCand()->eta() << "\t" << theLeadTrack->eta()
  << "\n LTPhi\t" << thePFTau.leadPFChargedHadrCand()->phi() << "\t" << theLeadTrack->phi()
  << "\n NIso\t" << thePFTau.isolationPFChargedHadrCands().size() << "\t" << theIsoTracks.size() - theSignalTracks.size()
  <<"\n";

  unsigned int nIsoTrk = 0;
  std::cout << " Tracks in L2.5 Iso: (Pt:Eta:Phi:DR:Chi2:Chi2/NdF:PxlHits:dxy:dz)\n";
  for(TrackRefVector::const_iterator isoIt = theIsoTracks.begin(); isoIt != theIsoTracks.end(); ++isoIt){
    double isoTrackLeadTrkDeltaR = deltaR(theLeadTrack->eta(), theLeadTrack->phi(), (*isoIt)->eta(), (*isoIt)->phi());
    if(isoTrackLeadTrkDeltaR > _signalCone){
      nIsoTrk++;
      cout << nIsoTrk 
           << "\t" << (*isoIt)->pt() 
	   << "\t" << (*isoIt)->eta() 
	   << "\t" << (*isoIt)->phi()
	   << "\t" << isoTrackLeadTrkDeltaR 
	   << "\t" << (*isoIt)->chi2()
	   << "\t" << (*isoIt)->normalizedChi2()
	   << "\t" << (*isoIt)->hitPattern().numberOfValidPixelHits()
	   << "\t" << (*isoIt)->dxy(theVertexPosition)
	   << "\t" << (*isoIt)->dz(theVertexPosition)
	   << "\t" << theVertexPosition
	   << "\n";
    }
  }
  nIsoTrk = 0;
  std::cout << "Tracks in PFTau Iso: (Pt:Eta:Phi)\n";
  for(PFCandidateRefVector::const_iterator isoIt = thePFTau.isolationPFChargedHadrCands().begin(); 
     isoIt != thePFTau.isolationPFChargedHadrCands().end(); ++isoIt){
    nIsoTrk++;
    cout << nIsoTrk << "\t" 
         << (*isoIt)->pt() << "\t" 
	 << (*isoIt)->eta() << "\t" 
	 << (*isoIt)->phi() << "\n";
  }
}

void L25TauAnalyzer::clearVectors(){
 l25TrkPt->clear();
 l25TrkEta->clear();
 l25TrkPhi->clear();
 l25TrkDz->clear();
 l25TrkDxy->clear();
 l25TrkChi2->clear();
 l25TrkChi2NdF->clear();
 l25TrkNRecHits->clear();
 l25TrkNValidPixelHits->clear();
 
 l25SignalTrkPt->clear();
 l25SignalTrkEta->clear();
 l25SignalTrkPhi->clear();
 l25SignalTrkDz->clear();
 l25SignalTrkDxy->clear();
 l25SignalTrkChi2NdF->clear();
 l25SignalTrkChi2->clear();
 l25SignalTrkNRecHits->clear();
 l25SignalTrkNValidHits->clear();
 l25SignalTrkNValidPixelHits->clear();
 l25SignalTrkNLostHits->clear();
 
 l25IsoTrkPt->clear();
 l25IsoTrkEta->clear();
 l25IsoTrkPhi->clear();
 l25IsoTrkDz->clear();
 l25IsoTrkDxy->clear();
 l25IsoTrkChi2->clear();
 l25IsoTrkChi2NdF->clear();
 l25IsoTrkNRecHits->clear();
 l25IsoTrkNValidHits->clear();
 l25IsoTrkNValidPixelHits->clear();
 l25IsoTrkNLostHits->clear();
}

void L25TauAnalyzer::initializeVectors(){
 l25TrkPt = NULL;
 l25TrkEta= NULL;
 l25TrkPhi= NULL;
 l25TrkDz= NULL;
 l25TrkDxy= NULL;
 l25TrkChi2= NULL;
 l25TrkChi2NdF= NULL;
 l25TrkNRecHits= NULL;
 l25TrkNValidPixelHits= NULL;
 
 l25SignalTrkPt= NULL;
 l25SignalTrkEta= NULL;
 l25SignalTrkPhi= NULL;
 l25SignalTrkDz= NULL;
 l25SignalTrkDxy= NULL;
 l25SignalTrkChi2= NULL;
 l25SignalTrkChi2NdF= NULL;
 l25SignalTrkNRecHits= NULL;
 l25SignalTrkNValidHits= NULL;
 l25SignalTrkNValidPixelHits= NULL;
 l25SignalTrkNLostHits= NULL;
 
 l25IsoTrkPt= NULL;
 l25IsoTrkChi2NdF= NULL;
 l25IsoTrkChi2= NULL;
 l25IsoTrkNValidHits= NULL;
 l25IsoTrkNRecHits= NULL;
 l25IsoTrkNValidPixelHits= NULL;
 l25IsoTrkNLostHits= NULL;
 l25IsoTrkDxy= NULL;
 l25IsoTrkDz= NULL;
 l25IsoTrkEta= NULL;
 l25IsoTrkPhi= NULL;
}

// ------------ method called once each job just before starting event loop  ------------
void L25TauAnalyzer::beginJob(){
}

// ------------ method called once each job just after ending the event loop  ------------
void L25TauAnalyzer::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(L25TauAnalyzer);
