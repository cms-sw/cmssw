/*  \class TurnOnMaker
 *
 *  Class to produce some turn on curves in the TriggerValidation Code
 *  the curves are produced by associating the reco and mc objects to l1 and hlt objects
 *
 *
 *  Author: Massimiliano Chiorboli      Date: November 2007
 //         Maurizio Pierini
 //         Maria Spiropulu
 *
 */
#include "HLTriggerOffline/SUSYBSM/interface/TurnOnMaker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTriggerOffline/SUSYBSM/interface/PtSorter.h"



using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;
using namespace l1extra;


TurnOnMaker::TurnOnMaker(edm::ParameterSet turnOn_params)
{
  m_genSrc             	 = turnOn_params.getParameter<string>("mcParticles");  
  m_recoMuonSrc    	 = turnOn_params.getParameter<string>("recoMuons");
  m_genMother            = turnOn_params.getParameter<string>("genMother");
  m_hltMuonTrackSrc      = turnOn_params.getParameter<string>("hltMuonTracks");
  m_hlt1MuonIsoSrc       = turnOn_params.getParameter<vstring>("hlt1MuonIsoList");
  m_hlt1MuonNonIsoSrc    = turnOn_params.getParameter<vstring>("hlt1MuonNonIsoList");

  theHLT1MuonIsoObjectVector.resize(m_hlt1MuonIsoSrc.size());
  theHLT1MuonNonIsoObjectVector.resize(m_hlt1MuonNonIsoSrc.size());

  s_Iso = "Iso";
  s_NonIso = "NonIso";

}


void TurnOnMaker::fillPlots(const edm::Event& iEvent)
{

  this->handleObjects(iEvent);

  
  // Distributions of Trigger Objects for the path HLT1MuonIso
  for(unsigned int i=0; i<theHLT1MuonIsoObjectVector.size(); i++) {
    hHLT1MuonIsoMult[i]      ->Fill(theHLT1MuonIsoObjectVector[i].size());
    for(unsigned int j=0; j<theHLT1MuonIsoObjectVector[i].size(); j++) {
      RecoChargedCandidateRef ref1 = RecoChargedCandidateRef(theHLT1MuonIsoObjectVector[i][j]);
      hHLT1MuonIsoPt[i]        ->Fill(ref1->pt() );	   	 	 
      hHLT1MuonIsoEta[i]       ->Fill(ref1->eta());	   	 	 
      hHLT1MuonIsoPhi[i]       ->Fill(ref1->phi());	   	  
    }
  }

  
  // Distributions of Trigger Objects for the path HLT1MuonNonIso
  for(unsigned int i=0; i<theHLT1MuonNonIsoObjectVector.size(); i++) {
    hHLT1MuonNonIsoMult[i]      ->Fill(theHLT1MuonNonIsoObjectVector[i].size());
    for(unsigned int j=0; j<theHLT1MuonNonIsoObjectVector[i].size(); j++) {
      RecoChargedCandidateRef ref1 = RecoChargedCandidateRef(theHLT1MuonNonIsoObjectVector[i][j]);
      hHLT1MuonNonIsoPt[i]        ->Fill(ref1->pt() );	   	 	 
      hHLT1MuonNonIsoEta[i]       ->Fill(ref1->eta());	   	 	 
      hHLT1MuonNonIsoPhi[i]       ->Fill(ref1->phi());	   	  
    }
  }


  // Distributions of the Muon Tracks used for the trigger (NOT reco muons)
  hMuonTrackMult->Fill(theMuonTrackCollection.size());
  for(unsigned int i=0; i<theMuonTrackCollection.size(); i++) {
    hMuonTrackPt          ->Fill(theMuonTrackCollection[i].pt());             
    hMuonTrackEta         ->Fill(theMuonTrackCollection[i].eta());             
    hMuonTrackPhi         ->Fill(theMuonTrackCollection[i].phi());
  }


  // Distributions of Reco Muons
  hRecoMuonMult->Fill(theMuonCollection.size());
  for(unsigned int i=0; i<theMuonCollection.size(); i++) {
    hRecoMuonPt          ->Fill(theMuonCollection[i].pt());             
    hRecoMuonEta         ->Fill(theMuonCollection[i].eta());             
    hRecoMuonPhi         ->Fill(theMuonCollection[i].phi());
    if(fabs(theMuonCollection[i].eta())<1.2) 
      {hRecoMuonPtBarrel->Fill(theMuonCollection[i].pt());}
    if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
      {hRecoMuonPtEndcap->Fill(theMuonCollection[i].pt());}
    if(theMuonCollection[i].pt()>10)
      {hRecoMuonEtaPt10->Fill(theMuonCollection[i].eta());}
    if(theMuonCollection[i].pt()>20)
      {hRecoMuonEtaPt20->Fill(theMuonCollection[i].eta());}
    
    //Association of Reco Muons with objects of HLT1MuonIso
    for(unsigned int j=0; j<theHLT1MuonIsoObjectVector.size(); j++) {
      if(recoToTriggerMatched((&theMuonCollection[i]),theHLT1MuonIsoObjectVector,j)) {
	hRecoMuonPtAssHLT1MuonIso[j]->Fill(theMuonCollection[i].pt());
	if(fabs(theMuonCollection[i].eta())<1.2) 
	  hRecoMuonPtAssHLT1MuonIsoBarrel[j]->Fill(theMuonCollection[i].pt());
	if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	  hRecoMuonPtAssHLT1MuonIsoEndcap[j]->Fill(theMuonCollection[i].pt());
	hRecoMuonEtaAssHLT1MuonIso[j]->Fill(theMuonCollection[i].eta());
	if(theMuonCollection[i].pt()>10) 
	  hRecoMuonEtaAssHLT1MuonIsoPt10[j]->Fill(theMuonCollection[i].eta());
	if(theMuonCollection[i].pt()>20) 
	  hRecoMuonEtaAssHLT1MuonIsoPt20[j]->Fill(theMuonCollection[i].eta());
      }
    }

   //Association of Reco Muons with objects of HLT1MuonNonIso
    for(unsigned int j=0; j<theHLT1MuonNonIsoObjectVector.size(); j++) {
      if(recoToTriggerMatched((&theMuonCollection[i]),theHLT1MuonNonIsoObjectVector,j)) {
	hRecoMuonPtAssHLT1MuonNonIso[j]->Fill(theMuonCollection[i].pt());
	if(fabs(theMuonCollection[i].eta())<1.2) 
	  hRecoMuonPtAssHLT1MuonNonIsoBarrel[j]->Fill(theMuonCollection[i].pt());
	if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	  hRecoMuonPtAssHLT1MuonNonIsoEndcap[j]->Fill(theMuonCollection[i].pt());
	hRecoMuonEtaAssHLT1MuonNonIso[j]->Fill(theMuonCollection[i].eta());
	if(theMuonCollection[i].pt()>10) 
	  hRecoMuonEtaAssHLT1MuonNonIsoPt10[j]->Fill(theMuonCollection[i].eta());
	if(theMuonCollection[i].pt()>20) 
	  hRecoMuonEtaAssHLT1MuonNonIsoPt20[j]->Fill(theMuonCollection[i].eta());
      }
    }


    // Association of RecoMuons with tracks used for trigger

    // Iso pt cut
    if(recoToTracksMatched(&theMuonCollection[i],theMuonTrackCollection,99999,s_Iso)) {
     
      hRecoMuonPtAssMuonTrackIso->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())<1.2) 
	hRecoMuonPtAssMuonTrackIsoBarrel->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	hRecoMuonPtAssMuonTrackIsoEndcap->Fill(theMuonCollection[i].pt());
      hRecoMuonEtaAssMuonTrackIso->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>10) 
	hRecoMuonEtaAssMuonTrackIsoPt10->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>20) 
	hRecoMuonEtaAssMuonTrackIsoPt20->Fill(theMuonCollection[i].eta());
    }

    if(recoToTracksMatched(&theMuonCollection[i],theMuonTrackCollection,2,s_Iso)) {
      hRecoMuonPtAssMuonTrackIsoDr2->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())<1.2) 
	hRecoMuonPtAssMuonTrackIsoDr2Barrel->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	hRecoMuonPtAssMuonTrackIsoDr2Endcap->Fill(theMuonCollection[i].pt());
      hRecoMuonEtaAssMuonTrackIsoDr2->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>10) 
	hRecoMuonEtaAssMuonTrackIsoDr2Pt10->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>20) 
	hRecoMuonEtaAssMuonTrackIsoDr2Pt20->Fill(theMuonCollection[i].eta());
    }


    if(recoToTracksMatched(&theMuonCollection[i],theMuonTrackCollection,0.2,s_Iso)) {
      hRecoMuonPtAssMuonTrackIsoDr02->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())<1.2) 
	hRecoMuonPtAssMuonTrackIsoDr02Barrel->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	hRecoMuonPtAssMuonTrackIsoDr02Endcap->Fill(theMuonCollection[i].pt());
      hRecoMuonEtaAssMuonTrackIsoDr02->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>10) 
	hRecoMuonEtaAssMuonTrackIsoDr02Pt10->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>20) 
	hRecoMuonEtaAssMuonTrackIsoDr02Pt20->Fill(theMuonCollection[i].eta());
    }

    if(recoToTracksMatched(&theMuonCollection[i],theMuonTrackCollection,0.02,s_Iso)) {
      hRecoMuonPtAssMuonTrackIsoDr002->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())<1.2) 
	hRecoMuonPtAssMuonTrackIsoDr002Barrel->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	hRecoMuonPtAssMuonTrackIsoDr002Endcap->Fill(theMuonCollection[i].pt());
      hRecoMuonEtaAssMuonTrackIsoDr002->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>10) 
	hRecoMuonEtaAssMuonTrackIsoDr002Pt10->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>20) 
	hRecoMuonEtaAssMuonTrackIsoDr002Pt20->Fill(theMuonCollection[i].eta());
    }



    // NonIso pt cut
    if(recoToTracksMatched(&theMuonCollection[i],theMuonTrackCollection,99999,s_NonIso)) {
      hRecoMuonPtAssMuonTrackNonIso->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())<1.2) 
	hRecoMuonPtAssMuonTrackNonIsoBarrel->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	hRecoMuonPtAssMuonTrackNonIsoEndcap->Fill(theMuonCollection[i].pt());
      hRecoMuonEtaAssMuonTrackNonIso->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>10) 
	hRecoMuonEtaAssMuonTrackNonIsoPt10->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>20) 
	hRecoMuonEtaAssMuonTrackNonIsoPt20->Fill(theMuonCollection[i].eta());
    }

    if(recoToTracksMatched(&theMuonCollection[i],theMuonTrackCollection,2,s_NonIso)) {
      hRecoMuonPtAssMuonTrackNonIsoDr2->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())<1.2) 
	hRecoMuonPtAssMuonTrackNonIsoDr2Barrel->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	hRecoMuonPtAssMuonTrackNonIsoDr2Endcap->Fill(theMuonCollection[i].pt());
      hRecoMuonEtaAssMuonTrackNonIsoDr2->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>10) 
	hRecoMuonEtaAssMuonTrackNonIsoDr2Pt10->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>20) 
	hRecoMuonEtaAssMuonTrackNonIsoDr2Pt20->Fill(theMuonCollection[i].eta());
    }

    if(recoToTracksMatched(&theMuonCollection[i],theMuonTrackCollection,0.2,s_NonIso)) {
      hRecoMuonPtAssMuonTrackNonIsoDr02->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())<1.2) 
	hRecoMuonPtAssMuonTrackNonIsoDr02Barrel->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	hRecoMuonPtAssMuonTrackNonIsoDr02Endcap->Fill(theMuonCollection[i].pt());
      hRecoMuonEtaAssMuonTrackNonIsoDr02->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>10) 
	hRecoMuonEtaAssMuonTrackNonIsoDr02Pt10->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>20) 
	hRecoMuonEtaAssMuonTrackNonIsoDr02Pt20->Fill(theMuonCollection[i].eta());
    }

    if(recoToTracksMatched(&theMuonCollection[i],theMuonTrackCollection,0.02,s_NonIso)) {
      hRecoMuonPtAssMuonTrackNonIsoDr002->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())<1.2) 
	hRecoMuonPtAssMuonTrackNonIsoDr002Barrel->Fill(theMuonCollection[i].pt());
      if(fabs(theMuonCollection[i].eta())>1.2 && fabs(theMuonCollection[i].eta())<2.1) 
	hRecoMuonPtAssMuonTrackNonIsoDr002Endcap->Fill(theMuonCollection[i].pt());
      hRecoMuonEtaAssMuonTrackNonIsoDr002->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>10) 
	hRecoMuonEtaAssMuonTrackNonIsoDr002Pt10->Fill(theMuonCollection[i].eta());
      if(theMuonCollection[i].pt()>20) 
	hRecoMuonEtaAssMuonTrackNonIsoDr002Pt20->Fill(theMuonCollection[i].eta());
    }



  }
      

  int nGenMuon = 0;
  for(unsigned int i=0; i<theGenParticleCollection->size(); i++) {
    const GenParticleCandidate* genParticle = dynamic_cast<const GenParticleCandidate*> (&(*theGenParticleCollection)[i]);
    if(genParticle->status() == 1) {
      if(fabs(genParticle->pdgId()) == 13) {
	bool isFromW    = false;
	bool isFromWtoJ = false;
	bool isFromB    = false;
	const GenParticleCandidate* genParticleMother = dynamic_cast<const GenParticleCandidate*> (genParticle->mother());
	while(abs(genParticleMother->pdgId()) == 13) {
	  genParticleMother = dynamic_cast<const GenParticleCandidate*> (genParticleMother->mother());
	}
	if(abs(genParticleMother->pdgId()) == 24) isFromW = true;
// 	cout <<"genParticleMother->pdgId()  = " << genParticleMother->pdgId() << endl;
// 	cout <<"genParticleMother->status() = " << genParticleMother->status() << endl;
// 	cout <<"isFromW = " << (int) isFromW << endl;

	while(genParticleMother->numberOfMothers()>0) {
	  genParticleMother = dynamic_cast<const GenParticleCandidate*> (genParticleMother->mother());
	  if(abs(genParticleMother->pdgId()) == 24) isFromWtoJ = true;
	  if(abs(genParticleMother->pdgId()) == 5)  isFromB    = true;
// 	  cout <<"meson chain genParticleMother->numberOfMothers() = " << genParticleMother->numberOfMothers() << endl;
// 	  cout <<"meson chain genParticleMother->pdgId()           = " << genParticleMother->pdgId()           << endl;
// 	  cout <<"meson chain genParticleMother->status()          = " << genParticleMother->status()          << endl;
// 	  cout <<"meson chain isFromW = " << (int) isFromW << endl;
// 	  cout <<"meson chain isFromB = " << (int) isFromB << endl;
	}

	bool condMuDecay;
	if(m_genMother == "W")         condMuDecay = isFromW;
	else if(m_genMother == "b")    condMuDecay = isFromB;
	else if(m_genMother == "WtoJ") condMuDecay = isFromWtoJ;
	else if(m_genMother == "All")  condMuDecay = true;
	else {
	  cout << "Wrong condition for the GenMuon mother" << endl;
	  cout << "All the GenMuons are selected" << endl;
	  condMuDecay = true;
	}
	
	if(condMuDecay) {

	  nGenMuon++;
	  hGenMuonPt          ->Fill(genParticle->pt());             
	  hGenMuonEta         ->Fill(genParticle->eta());             
	  hGenMuonPhi         ->Fill(genParticle->phi());
	  if(fabs(genParticle->eta())<1.2) 
	    {hGenMuonPtBarrel->Fill(genParticle->pt());}
	  if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	    {hGenMuonPtEndcap->Fill(genParticle->pt());}
	  if(genParticle->pt()>10)
	    {hGenMuonEtaPt10->Fill(genParticle->eta());}
	  if(genParticle->pt()>20)
	    {hGenMuonEtaPt20->Fill(genParticle->eta());}
	
	  //Association of Gen Muons with objects of HLT1MuonIso
	  for(unsigned int j=0; j<theHLT1MuonIsoObjectVector.size(); j++) {
	    if(recoToTriggerMatched(&(*theGenParticleCollection)[i],theHLT1MuonIsoObjectVector,j)) {
	      hGenMuonPtAssHLT1MuonIso[j]->Fill(genParticle->pt());
	      if(fabs(genParticle->eta())<1.2) 
		hGenMuonPtAssHLT1MuonIsoBarrel[j]->Fill(genParticle->pt());
	      if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
		hGenMuonPtAssHLT1MuonIsoEndcap[j]->Fill(genParticle->pt());
	      hGenMuonEtaAssHLT1MuonIso[j]->Fill(genParticle->eta());
	      if(genParticle->pt()>10) 
		hGenMuonEtaAssHLT1MuonIsoPt10[j]->Fill(genParticle->eta());
	      if(genParticle->pt()>20) 
		hGenMuonEtaAssHLT1MuonIsoPt20[j]->Fill(genParticle->eta());
	    }
	  }
	
	  //Association of Gen Muons with objects of HLT1MuonNonIso
	  for(unsigned int j=0; j<theHLT1MuonNonIsoObjectVector.size(); j++) {
	    if(recoToTriggerMatched(&(*theGenParticleCollection)[i],theHLT1MuonNonIsoObjectVector,j)) {
	      hGenMuonPtAssHLT1MuonNonIso[j]->Fill(genParticle->pt());
	      if(fabs(genParticle->eta())<1.2) 
		hGenMuonPtAssHLT1MuonNonIsoBarrel[j]->Fill(genParticle->pt());
	      if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
		hGenMuonPtAssHLT1MuonNonIsoEndcap[j]->Fill(genParticle->pt());
	      hGenMuonEtaAssHLT1MuonNonIso[j]->Fill(genParticle->eta());
	      if(genParticle->pt()>10) 
		hGenMuonEtaAssHLT1MuonNonIsoPt10[j]->Fill(genParticle->eta());
	      if(genParticle->pt()>20) 
		hGenMuonEtaAssHLT1MuonNonIsoPt20[j]->Fill(genParticle->eta());
	    }
	  }

	  // Association of GenMuons with tracks used for trigger

	  // Iso pt cut
	  if(recoToTracksMatched(&(*theGenParticleCollection)[i],theMuonTrackCollection,99999,s_Iso)) {
	    hGenMuonPtAssMuonTrackIso->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())<1.2) 
	      hGenMuonPtAssMuonTrackIsoBarrel->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	      hGenMuonPtAssMuonTrackIsoEndcap->Fill(genParticle->pt());
	    hGenMuonEtaAssMuonTrackIso->Fill(genParticle->eta());
	    if(genParticle->pt()>10) 
	      hGenMuonEtaAssMuonTrackIsoPt10->Fill(genParticle->eta());
	    if(genParticle->pt()>20) 
	      hGenMuonEtaAssMuonTrackIsoPt20->Fill(genParticle->eta());
	  }

	  if(recoToTracksMatched(&(*theGenParticleCollection)[i],theMuonTrackCollection,2,s_Iso)) {
	    hGenMuonPtAssMuonTrackIsoDr2->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())<1.2) 
	      hGenMuonPtAssMuonTrackIsoDr2Barrel->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	      hGenMuonPtAssMuonTrackIsoDr2Endcap->Fill(genParticle->pt());
	    hGenMuonEtaAssMuonTrackIsoDr2->Fill(genParticle->eta());
	    if(genParticle->pt()>10) 
	      hGenMuonEtaAssMuonTrackIsoDr2Pt10->Fill(genParticle->eta());
	    if(genParticle->pt()>20) 
	      hGenMuonEtaAssMuonTrackIsoDr2Pt20->Fill(genParticle->eta());
	  }

	  if(recoToTracksMatched(&(*theGenParticleCollection)[i],theMuonTrackCollection,0.2,s_Iso)) {
	    hGenMuonPtAssMuonTrackIsoDr02->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())<1.2) 
	      hGenMuonPtAssMuonTrackIsoDr02Barrel->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	      hGenMuonPtAssMuonTrackIsoDr02Endcap->Fill(genParticle->pt());
	    hGenMuonEtaAssMuonTrackIsoDr02->Fill(genParticle->eta());
	    if(genParticle->pt()>10) 
	      hGenMuonEtaAssMuonTrackIsoDr02Pt10->Fill(genParticle->eta());
	    if(genParticle->pt()>20) 
	      hGenMuonEtaAssMuonTrackIsoDr02Pt20->Fill(genParticle->eta());
	  }

	  if(recoToTracksMatched(&(*theGenParticleCollection)[i],theMuonTrackCollection,0.02,s_Iso)) {
	    hGenMuonPtAssMuonTrackIsoDr002->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())<1.2) 
	      hGenMuonPtAssMuonTrackIsoDr002Barrel->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	      hGenMuonPtAssMuonTrackIsoDr002Endcap->Fill(genParticle->pt());
	    hGenMuonEtaAssMuonTrackIsoDr002->Fill(genParticle->eta());
	    if(genParticle->pt()>10) 
	      hGenMuonEtaAssMuonTrackIsoDr002Pt10->Fill(genParticle->eta());
	    if(genParticle->pt()>20) 
	      hGenMuonEtaAssMuonTrackIsoDr002Pt20->Fill(genParticle->eta());
	  }


	  // NonIso pt cut
	  if(recoToTracksMatched(&(*theGenParticleCollection)[i],theMuonTrackCollection,99999,s_NonIso)) {
	    hGenMuonPtAssMuonTrackNonIso->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())<1.2) 
	      hGenMuonPtAssMuonTrackNonIsoBarrel->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	      hGenMuonPtAssMuonTrackNonIsoEndcap->Fill(genParticle->pt());
	    hGenMuonEtaAssMuonTrackNonIso->Fill(genParticle->eta());
	    if(genParticle->pt()>10) 
	      hGenMuonEtaAssMuonTrackNonIsoPt10->Fill(genParticle->eta());
	    if(genParticle->pt()>20) 
	      hGenMuonEtaAssMuonTrackNonIsoPt20->Fill(genParticle->eta());
	  }

	  if(recoToTracksMatched(&(*theGenParticleCollection)[i],theMuonTrackCollection,2,s_NonIso)) {
	    hGenMuonPtAssMuonTrackNonIsoDr2->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())<1.2) 
	      hGenMuonPtAssMuonTrackNonIsoDr2Barrel->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	      hGenMuonPtAssMuonTrackNonIsoDr2Endcap->Fill(genParticle->pt());
	    hGenMuonEtaAssMuonTrackNonIsoDr2->Fill(genParticle->eta());
	    if(genParticle->pt()>10) 
	      hGenMuonEtaAssMuonTrackNonIsoDr2Pt10->Fill(genParticle->eta());
	    if(genParticle->pt()>20) 
	      hGenMuonEtaAssMuonTrackNonIsoDr2Pt20->Fill(genParticle->eta());
	  }

	  if(recoToTracksMatched(&(*theGenParticleCollection)[i],theMuonTrackCollection,0.2,s_NonIso)) {
	    hGenMuonPtAssMuonTrackNonIsoDr02->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())<1.2) 
	      hGenMuonPtAssMuonTrackNonIsoDr02Barrel->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	      hGenMuonPtAssMuonTrackNonIsoDr02Endcap->Fill(genParticle->pt());
	    hGenMuonEtaAssMuonTrackNonIsoDr02->Fill(genParticle->eta());
	    if(genParticle->pt()>10) 
	      hGenMuonEtaAssMuonTrackNonIsoDr02Pt10->Fill(genParticle->eta());
	    if(genParticle->pt()>20) 
	      hGenMuonEtaAssMuonTrackNonIsoDr02Pt20->Fill(genParticle->eta());
	  }

	  if(recoToTracksMatched(&(*theGenParticleCollection)[i],theMuonTrackCollection,0.02,s_NonIso)) {
	    hGenMuonPtAssMuonTrackNonIsoDr002->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())<1.2) 
	      hGenMuonPtAssMuonTrackNonIsoDr002Barrel->Fill(genParticle->pt());
	    if(fabs(genParticle->eta())>1.2 && fabs(genParticle->eta())<2.1) 
	      hGenMuonPtAssMuonTrackNonIsoDr002Endcap->Fill(genParticle->pt());
	    hGenMuonEtaAssMuonTrackNonIsoDr002->Fill(genParticle->eta());
	    if(genParticle->pt()>10) 
	      hGenMuonEtaAssMuonTrackNonIsoDr002Pt10->Fill(genParticle->eta());
	    if(genParticle->pt()>20) 
	      hGenMuonEtaAssMuonTrackNonIsoDr002Pt20->Fill(genParticle->eta());
	  }



	
	}


      }
    }
  }
  hGenMuonMult->Fill(nGenMuon);
  
  //   cout << "-----------------------------------" << endl;
//   cout << "HLT path HLT1MuonIso: " << endl;
//   cout << endl;
//   for(unsigned int i=0; i<theHLT1MuonIsoObjectVector.size(); i++) {
//     cout << "Trigger object " << m_hlt1MuonIsoSrc[i] << endl;
//     cout << "theHLT1MuonIsoObjectVector[i].size() = " << theHLT1MuonIsoObjectVector[i].size() << endl;
//     for(unsigned int j=0; j<theHLT1MuonIsoObjectVector[i].size(); j++) {
//       edm::RefToBase<reco::Candidate> ref1 = theHLT1MuonIsoObjectVector[i][j];
//       cout << "theHLT1MuonIsoObjectVector[" << i << "]" << j << "].pt()  = " << ref1->pt()  << endl;
//       cout << "theHLT1MuonIsoObjectVector[" << i << "]" << j << "].eta() = " << ref1->eta() << endl;
//       cout << "theHLT1MuonIsoObjectVector[" << i << "]" << j << "].phi() = " << ref1->phi() << endl;
//     }
//   }
//   cout << "-----------------------------------" << endl;


//   cout << "-----------------------------------" << endl;
//   cout << "HLT path HLT1MuonNonIso: " << endl;
//   cout << endl;
//   for(unsigned int i=0; i<theHLT1MuonNonIsoObjectVector.size(); i++) {
//     cout << "Trigger object " << m_hlt1MuonNonIsoSrc[i] << endl;
//     cout << "theHLT1MuonNonIsoObjectVector[i].size() = " << theHLT1MuonNonIsoObjectVector[i].size() << endl;
//     for(unsigned int j=0; j<theHLT1MuonNonIsoObjectVector[i].size(); j++) {
//       edm::RefToBase<reco::Candidate> ref1 = theHLT1MuonNonIsoObjectVector[i][j];
//       cout << "theHLT1MuonNonIsoObjectVector[" << i << "]" << j << "].pt()  = " << ref1->pt()  << endl;
//       cout << "theHLT1MuonNonIsoObjectVector[" << i << "]" << j << "].eta() = " << ref1->eta() << endl;
//       cout << "theHLT1MuonNonIsoObjectVector[" << i << "]" << j << "].phi() = " << ref1->phi() << endl;
//     }
//   }
//   cout << "-----------------------------------" << endl;
    





//     cout <<"theMuonCollection.size() = " << theMuonCollection.size() << endl;
//     for(unsigned int i=0; i<theMuonCollection.size(); i++) {
//       cout << "theMuonCollection["<< i << "].pt()   = " << theMuonCollection[i].pt() << endl;
//       cout << "theMuonCollection["<< i << "].eta()  = " << theMuonCollection[i].eta() << endl;
//       cout << "theMuonCollection["<< i << "].phi()  = " << theMuonCollection[i].phi() << endl;
//       for(unsigned int j=0; j<theHLT1MuonIsoObjectVector.size(); j++) {
// 	cout << "Matching for HLT1MuonIso " << endl;
// 	cout << "theMuonCollection["<< i << "] matching with HLT level " << j << " = " << (int) recoToTriggerMatched((&theMuonCollection[i]), theHLT1MuonIsoObjectVector, j) << endl;
//       }
//       for(unsigned int j=0; j<theHLT1MuonNonIsoObjectVector.size(); j++) {
// 	cout << "Matching for HLT1MuonNonIso " << endl;
// 	cout << "theMuonCollection["<< i << "] matching with HLT level " << j << " = " << (int) recoToTriggerMatched((&theMuonCollection[i]), theHLT1MuonNonIsoObjectVector, j) << endl;
//       }
//     }



}


void
TurnOnMaker::writeHistos() {
  gDirectory->cd("/TurnOnCurves/Muon");
  for(unsigned int i=0; i<hHLT1MuonIsoMult.size(); i++) {
    hHLT1MuonIsoMult[i]->Write();
    hHLT1MuonIsoPt[i]->Write();
    hHLT1MuonIsoEta[i]->Write();
    hHLT1MuonIsoPhi[i]->Write();
  }

  for(unsigned int i=0; i<hHLT1MuonNonIsoMult.size(); i++) {
    hHLT1MuonNonIsoMult[i]->Write();
    hHLT1MuonNonIsoPt[i]->Write();
    hHLT1MuonNonIsoEta[i]->Write();
    hHLT1MuonNonIsoPhi[i]->Write();
  }  


  hMuonTrackPt->Write();
  hMuonTrackEta->Write();
  hMuonTrackPhi->Write();
  hMuonTrackMult->Write();

  hRecoMuonPt->Write();            
  hRecoMuonEta->Write();           
  hRecoMuonPhi->Write();           
  hRecoMuonMult->Write();          

  hRecoMuonPtBarrel->Write();       
  hRecoMuonPtEndcap->Write();       

  hRecoMuonEtaPt10->Write();       
  hRecoMuonEtaPt20->Write();       


  for(unsigned int i=0; i<hRecoMuonPtAssHLT1MuonIso.size(); i++) {
    hRecoMuonPtAssHLT1MuonIso[i]->Write();
    hRecoMuonPtAssHLT1MuonIsoBarrel[i]->Write();
    hRecoMuonPtAssHLT1MuonIsoEndcap[i]->Write();
    hRecoMuonEtaAssHLT1MuonIso[i]->Write();
    hRecoMuonEtaAssHLT1MuonIsoPt10[i]->Write();
    hRecoMuonEtaAssHLT1MuonIsoPt20[i]->Write();
  }    

  for(unsigned int i=0; i<hRecoMuonPtAssHLT1MuonNonIso.size(); i++) {
    hRecoMuonPtAssHLT1MuonNonIso[i]->Write();
    hRecoMuonPtAssHLT1MuonNonIsoBarrel[i]->Write();
    hRecoMuonPtAssHLT1MuonNonIsoEndcap[i]->Write();
    hRecoMuonEtaAssHLT1MuonNonIso[i]->Write();
    hRecoMuonEtaAssHLT1MuonNonIsoPt10[i]->Write();
    hRecoMuonEtaAssHLT1MuonNonIsoPt20[i]->Write();
  }  

  // Pt dirtibutions of Reco Muons associated
  // to Muon tracks used to build the trigger
  hRecoMuonPtAssMuonTrackIso->Write();
  hRecoMuonPtAssMuonTrackIsoBarrel->Write();
  hRecoMuonPtAssMuonTrackIsoEndcap->Write();
  hRecoMuonPtAssMuonTrackIsoDr2->Write();
  hRecoMuonPtAssMuonTrackIsoDr2Barrel->Write();
  hRecoMuonPtAssMuonTrackIsoDr2Endcap->Write();
  hRecoMuonPtAssMuonTrackIsoDr02->Write();
  hRecoMuonPtAssMuonTrackIsoDr02Barrel->Write();
  hRecoMuonPtAssMuonTrackIsoDr02Endcap->Write();
  hRecoMuonPtAssMuonTrackIsoDr002->Write();
  hRecoMuonPtAssMuonTrackIsoDr002Barrel->Write();
  hRecoMuonPtAssMuonTrackIsoDr002Endcap->Write();

  hRecoMuonPtAssMuonTrackNonIso->Write();
  hRecoMuonPtAssMuonTrackNonIsoBarrel->Write();
  hRecoMuonPtAssMuonTrackNonIsoEndcap->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr2->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr2Barrel->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr2Endcap->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr02->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr02Barrel->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr02Endcap->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr002->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr002Barrel->Write();
  hRecoMuonPtAssMuonTrackNonIsoDr002Endcap->Write();

  // Eta dirtibutions of Reco Muons associated
  // to Muon tracks used to build the trigger
  hRecoMuonEtaAssMuonTrackIso->Write();
  hRecoMuonEtaAssMuonTrackIsoPt10->Write();
  hRecoMuonEtaAssMuonTrackIsoPt20->Write();
  hRecoMuonEtaAssMuonTrackIsoDr2->Write();
  hRecoMuonEtaAssMuonTrackIsoDr2Pt10->Write();
  hRecoMuonEtaAssMuonTrackIsoDr2Pt20->Write();
  hRecoMuonEtaAssMuonTrackIsoDr02->Write();
  hRecoMuonEtaAssMuonTrackIsoDr02Pt10->Write();
  hRecoMuonEtaAssMuonTrackIsoDr02Pt20->Write();
  hRecoMuonEtaAssMuonTrackIsoDr002->Write();
  hRecoMuonEtaAssMuonTrackIsoDr002Pt10->Write();
  hRecoMuonEtaAssMuonTrackIsoDr002Pt20->Write();

  hRecoMuonEtaAssMuonTrackNonIso->Write();
  hRecoMuonEtaAssMuonTrackNonIsoPt10->Write();
  hRecoMuonEtaAssMuonTrackNonIsoPt20->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr2->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr2Pt10->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr2Pt20->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr02->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr02Pt10->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr02Pt20->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr002->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr002Pt10->Write();
  hRecoMuonEtaAssMuonTrackNonIsoDr002Pt20->Write();



  //Distributions of the Gen Muons
  hGenMuonPt->Write();             
  hGenMuonEta->Write();            
  hGenMuonPhi->Write();            
  hGenMuonMult->Write();           


  hGenMuonPtBarrel->Write();        
  hGenMuonPtEndcap->Write();        

  hGenMuonEtaPt10->Write();        
  hGenMuonEtaPt20->Write();        


  for(unsigned int i=0; i<hGenMuonPtAssHLT1MuonIso.size(); i++) {
    hGenMuonPtAssHLT1MuonIso[i]->Write();
    hGenMuonPtAssHLT1MuonIsoBarrel[i]->Write();
    hGenMuonPtAssHLT1MuonIsoEndcap[i]->Write();
    hGenMuonEtaAssHLT1MuonIso[i]->Write();
    hGenMuonEtaAssHLT1MuonIsoPt10[i]->Write();
    hGenMuonEtaAssHLT1MuonIsoPt20[i]->Write();
  }


  
  for(unsigned int i=0; i<hGenMuonPtAssHLT1MuonNonIso.size(); i++) {
    hGenMuonPtAssHLT1MuonNonIso[i]->Write();
    hGenMuonPtAssHLT1MuonNonIsoBarrel[i]->Write();
    hGenMuonPtAssHLT1MuonNonIsoEndcap[i]->Write();
    hGenMuonEtaAssHLT1MuonNonIso[i]->Write();
    hGenMuonEtaAssHLT1MuonNonIsoPt10[i]->Write();
    hGenMuonEtaAssHLT1MuonNonIsoPt20[i]->Write();
  }


  
  // Pt dirtibutions of Gen Muons associated
  // to Muon tracks used to build the trigger
  hGenMuonPtAssMuonTrackIso->Write();
  hGenMuonPtAssMuonTrackIsoBarrel->Write();
  hGenMuonPtAssMuonTrackIsoEndcap->Write();
  hGenMuonPtAssMuonTrackIsoDr2->Write();
  hGenMuonPtAssMuonTrackIsoDr2Barrel->Write();
  hGenMuonPtAssMuonTrackIsoDr2Endcap->Write();
  hGenMuonPtAssMuonTrackIsoDr02->Write();
  hGenMuonPtAssMuonTrackIsoDr02Barrel->Write();
  hGenMuonPtAssMuonTrackIsoDr02Endcap->Write();
  hGenMuonPtAssMuonTrackIsoDr002->Write();
  hGenMuonPtAssMuonTrackIsoDr002Barrel->Write();
  hGenMuonPtAssMuonTrackIsoDr002Endcap->Write();

  hGenMuonPtAssMuonTrackNonIso->Write();
  hGenMuonPtAssMuonTrackNonIsoBarrel->Write();
  hGenMuonPtAssMuonTrackNonIsoEndcap->Write();
  hGenMuonPtAssMuonTrackNonIsoDr2->Write();
  hGenMuonPtAssMuonTrackNonIsoDr2Barrel->Write();
  hGenMuonPtAssMuonTrackNonIsoDr2Endcap->Write();
  hGenMuonPtAssMuonTrackNonIsoDr02->Write();
  hGenMuonPtAssMuonTrackNonIsoDr02Barrel->Write();
  hGenMuonPtAssMuonTrackNonIsoDr02Endcap->Write();
  hGenMuonPtAssMuonTrackNonIsoDr002->Write();
  hGenMuonPtAssMuonTrackNonIsoDr002Barrel->Write();
  hGenMuonPtAssMuonTrackNonIsoDr002Endcap->Write();

  // Eta dirtibutions of Gen Muons associated
  // to Muon tracks used to build the trigger
  hGenMuonEtaAssMuonTrackIso->Write();
  hGenMuonEtaAssMuonTrackIsoPt10->Write();
  hGenMuonEtaAssMuonTrackIsoPt20->Write();
  hGenMuonEtaAssMuonTrackIsoDr2->Write();
  hGenMuonEtaAssMuonTrackIsoDr2Pt10->Write();
  hGenMuonEtaAssMuonTrackIsoDr2Pt20->Write();
  hGenMuonEtaAssMuonTrackIsoDr02->Write();
  hGenMuonEtaAssMuonTrackIsoDr02Pt10->Write();
  hGenMuonEtaAssMuonTrackIsoDr02Pt20->Write();
  hGenMuonEtaAssMuonTrackIsoDr002->Write();
  hGenMuonEtaAssMuonTrackIsoDr002Pt10->Write();
  hGenMuonEtaAssMuonTrackIsoDr002Pt20->Write();

  hGenMuonEtaAssMuonTrackNonIso->Write();
  hGenMuonEtaAssMuonTrackNonIsoPt10->Write();
  hGenMuonEtaAssMuonTrackNonIsoPt20->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr2->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr2Pt10->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr2Pt20->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr02->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr02Pt10->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr02Pt20->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr002->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr002Pt10->Write();
  hGenMuonEtaAssMuonTrackNonIsoDr002Pt20->Write();





}

// void TurnOnMaker::finalOperations() {

//   //   int nBin = hRecoMuonPt->GetNbinsX();
//   //   std::vector<double> errorsA;
//   //   std::vector<double> errorsB;
//   //   std::vector<double> errors;
//   //   for(int i=0; i<nBin; i++) {
//   //     errorsA.push_back(hRecoMuonHltIsoMuonPt->GetBinError(i+1));
//   //     errorsB.push_back(hRecoMuonPt->GetBinError(i+1));
//   //     if(hRecoMuonPt->GetBinContent(i+1) == 0) {errors.push_back(0);}
//   //     else {
//   //     errors.push_back(sqrt( 
//   // 		     1/pow(hRecoMuonPt->GetBinContent(i+1),2) * pow(errorsA[i],2) +
//   // 		     pow(hRecoMuonHltIsoMuonPt->GetBinContent(i+1),2)/pow(hRecoMuonPt->GetBinContent(i+1),4) * pow(errorsB[i],2)
//   // 		     ));
//   //     }
//   //     cout << "errors[" << i << "] = " << errors[i] << endl;
//   //   }

//   hTurnOnRecoMuonVsRecoHltIsoMuon->Divide(hRecoMuonHltIsoMuonPt,hRecoMuonPt);
//   //   for(int i=0; i<nBin; i++) 
//   //     {cout << "err_div[" << i << "] = " << hTurnOnRecoMuonVsRecoHltIsoMuon->GetBinError(i+1) << endl;}
//   hTurnOnRecoMuonVsRecoL1IsoMuon ->Divide(hRecoMuonL1IsoMuonPt,hRecoMuonPt);
//   hTurnOnRecoHltIsoMuonVsRecoL1IsoMuon->Divide(hRecoMuonHltIsoMuonPt,hRecoMuonL1IsoMuonPt);


//   hTurnOnGenMuonVsGenHltIsoMuon->Divide(hGenMuonHltIsoMuonPt,hGenMuonPt);          
//   hTurnOnGenMuonVsGenL1IsoMuon->Divide(hGenMuonL1IsoMuonPt,hGenMuonPt);           
//   hTurnOnGenHltIsoMuonVsGenL1IsoMuon->Divide(hGenMuonHltIsoMuonPt,hGenMuonL1IsoMuonPt);  


//   hTurnOnRecoMuonVsRecoHltIsoMuonBarrel ->Divide(hRecoMuonHltIsoMuonPtBarrel,hRecoMuonPtBarrel);
//   hTurnOnRecoMuonVsRecoL1IsoMuonBarrel  ->Divide(hRecoMuonL1IsoMuonPtBarrel,hRecoMuonPtBarrel);
//   hTurnOnRecoHltIsoMuonVsRecoL1IsoMuonBarrel->Divide(hRecoMuonHltIsoMuonPtBarrel,hRecoMuonL1IsoMuonPtBarrel);

//   hTurnOnGenMuonVsGenHltIsoMuonBarrel        ->Divide(hGenMuonHltIsoMuonPtBarrel,hGenMuonPtBarrel);          
//   hTurnOnGenMuonVsGenL1IsoMuonBarrel         ->Divide(hGenMuonL1IsoMuonPtBarrel,hGenMuonPtBarrel);           
//   hTurnOnGenHltIsoMuonVsGenL1IsoMuonBarrel->Divide(hGenMuonHltIsoMuonPtBarrel,hGenMuonL1IsoMuonPtBarrel);  



//   hTurnOnRecoMuonVsRecoHltIsoMuonEndcap  ->Divide(hRecoMuonHltIsoMuonPtEndcap,hRecoMuonPtEndcap);
//   hTurnOnRecoMuonVsRecoL1IsoMuonEndcap   ->Divide(hRecoMuonL1IsoMuonPtEndcap,hRecoMuonPtEndcap);
//   hTurnOnRecoHltIsoMuonVsRecoL1IsoMuonEndcap ->Divide(hRecoMuonHltIsoMuonPtEndcap,hRecoMuonL1IsoMuonPtEndcap);

//   hTurnOnGenMuonVsGenHltIsoMuonEndcap        ->Divide(hGenMuonHltIsoMuonPtEndcap,hGenMuonPtEndcap);          
//   hTurnOnGenMuonVsGenL1IsoMuonEndcap         ->Divide(hGenMuonL1IsoMuonPtEndcap,hGenMuonPtEndcap);           
//   hTurnOnGenHltIsoMuonVsGenL1IsoMuonEndcap->Divide(hGenMuonHltIsoMuonPtEndcap,hGenMuonL1IsoMuonPtEndcap);  

// }



void TurnOnMaker::bookHistos() {


  gDirectory->cd("/TurnOnCurves/Muon");

  // book histos of Hlt objects for path HLT1MuonIso
  for(unsigned int i=0; i<m_hlt1MuonIsoSrc.size(); i++) {
    myHistoName = "HLT1MuonIsoMult_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "HLT1MuonIso object multiplicity for " + m_hlt1MuonIsoSrc[i];
    hHLT1MuonIsoMult.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10 ,  0   , 10  ));
    myHistoName = "HLT1MuonIsoPt_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "HLT1MuonIso object pt for " + m_hlt1MuonIsoSrc[i];
    hHLT1MuonIsoPt.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str()  , 50 ,  0   , 50  ));
    myHistoName = "HLT1MuonIsoEta_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "HLT1MuonIso object eta for " + m_hlt1MuonIsoSrc[i];
    hHLT1MuonIsoEta.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 , -3   , 3   ));
    myHistoName = "HLT1MuonIsoPhi_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "HLT1MuonIso object phi for " + m_hlt1MuonIsoSrc[i];
    hHLT1MuonIsoPhi.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 , -3.2 , 3.2 ));
  } 

  // book histos of Hlt objects for path HLT1MuonNonIso
  for(unsigned int i=0; i<m_hlt1MuonNonIsoSrc.size(); i++) {
    myHistoName = "HLT1MuonNonIsoMult_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "HLT1MuonNonIso object multiplicity for " + m_hlt1MuonNonIsoSrc[i];
    hHLT1MuonNonIsoMult.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 10 ,  0   , 10  ));
    myHistoName = "HLT1MuonNonIsoPt_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "HLT1MuonNonIso object pt for " + m_hlt1MuonNonIsoSrc[i];
    hHLT1MuonNonIsoPt.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str()  , 50 ,  0   , 50  ));
    myHistoName = "HLT1MuonNonIsoEta_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "HLT1MuonNonIso object eta for " + m_hlt1MuonNonIsoSrc[i];
    hHLT1MuonNonIsoEta.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 , -3   , 3   ));
    myHistoName = "HLT1MuonNonIsoPhi_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "HLT1MuonNonIso object phi for " + m_hlt1MuonNonIsoSrc[i];
    hHLT1MuonNonIsoPhi.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 , -3.2 , 3.2 ));
  } 


  // book histos for Muon tracks used to build the trigger (NOT reco muons)
  hMuonTrackPt   = new TH1D("MuonTrackPt"   ,"MuonTrackPt"   , 50 ,  0   , 50  );        
  hMuonTrackEta  = new TH1D("MuonTrackEta"  ,"MuonTrackEta"  , 50 ,  -3   , 3  );        
  hMuonTrackPhi  = new TH1D("MuonTrackPhi"  ,"MuonTrackPhi"  , 50 ,  -3.5   , 3.5  );    
  hMuonTrackMult = new TH1D("MuonTrackMult" ,"MuonTrackMult" , 10 ,  0   , 10  );        







  // book histos for reco muons
  hRecoMuonPt                    = new TH1D("RecoMuonPt"                  , "RecoMuonPt                      " , 50 ,  0   , 50  );                   
  hRecoMuonEta                   = new TH1D("RecoMuonEta"                  , "RecoMuonEta                    " , 50 ,  -3   , 3  );                   
  hRecoMuonPhi                   = new TH1D("RecoMuonPhi"                  , "RecoMuonPhi                    " , 50 ,  -3.5   , 3.5  );                   
  hRecoMuonMult                  = new TH1D("RecoMuonMult"                  , "RecoMuonMult                  " , 10 ,  0   , 10  );                   

  hRecoMuonPtBarrel              = new TH1D("RecoMuonPtBarrel"             , "RecoMuonPtBarrel             " , 50 ,  0   , 50  );                   
  hRecoMuonPtEndcap              = new TH1D("RecoMuonPtEndcap"             , "RecoMuonPtEndcap             " , 50 ,  0   , 50  );                   

  hRecoMuonEtaPt10               = new TH1D("RecoMuonEtaPt10"              , "RecoMuonEtaPt10              " , 50 ,  -3   , 3  );                   
  hRecoMuonEtaPt20               = new TH1D("RecoMuonEtaPt20"              , "RecoMuonEtaPt20              " , 50 ,  -3   , 3  );                   


  // book histos for reco muons associated to HLT objects for path HLT1MuonIso
  for(unsigned int i=0; i<m_hlt1MuonIsoSrc.size(); i++) {
    myHistoName  = "RecoMuonPtAssHLT1MuonIso_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Pt of RecoMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hRecoMuonPtAssHLT1MuonIso.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
   myHistoName  = "RecoMuonPtAssHLT1MuonIsoBarrel_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Barrel Pt of RecoMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hRecoMuonPtAssHLT1MuonIsoBarrel.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "RecoMuonPtAssHLT1MuonIsoEndcap_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Endcap Pt of RecoMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hRecoMuonPtAssHLT1MuonIsoEndcap.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "RecoMuonEtaAssHLT1MuonIso_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Eta of RecoMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hRecoMuonEtaAssHLT1MuonIso.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));    
   myHistoName  = "RecoMuonEtaAssHLT1MuonIsoPt10_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Pt>10 Eta of RecoMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hRecoMuonEtaAssHLT1MuonIsoPt10.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));
     myHistoName  = "RecoMuonEtaAssHLT1MuonIsoPt20_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Pt>20 Eta of RecoMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hRecoMuonEtaAssHLT1MuonIsoPt20.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));
   }

  // book histos for reco muons associated to HLT objects for path HLT1MuonNonIso
  for(unsigned int i=0; i<m_hlt1MuonNonIsoSrc.size(); i++) {
    myHistoName  = "RecoMuonPtAssHLT1MuonNonIso_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Pt of RecoMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hRecoMuonPtAssHLT1MuonNonIso.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "RecoMuonPtAssHLT1MuonNonIsoBarrel_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Barrel Pt of RecoMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hRecoMuonPtAssHLT1MuonNonIsoBarrel.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "RecoMuonPtAssHLT1MuonNonIsoEndcap_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Endcap Pt of RecoMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hRecoMuonPtAssHLT1MuonNonIsoEndcap.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "RecoMuonEtaAssHLT1MuonNonIso_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Eta of RecoMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hRecoMuonEtaAssHLT1MuonNonIso.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));    
    myHistoName  = "RecoMuonEtaAssHLT1MuonNonIsoPt10_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Pt>10 Eta of RecoMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hRecoMuonEtaAssHLT1MuonNonIsoPt10.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));
    myHistoName  = "RecoMuonEtaAssHLT1MuonNonIsoPt20_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Pt>20 Eta of RecoMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hRecoMuonEtaAssHLT1MuonNonIsoPt20.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));
  }



  // book histos for reco muons associated to muon tracks used for trigger
  hRecoMuonPtAssMuonTrackIso            = new TH1D("RecoMuonPtAssMuonTrackIso"            , "RecoMuonPtAssMuonTrackIso            " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoBarrel      = new TH1D("RecoMuonPtAssMuonTrackIsoBarrel"      , "RecoMuonPtAssMuonTrackIsoBarrel      " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoEndcap      = new TH1D("RecoMuonPtAssMuonTrackIsoEndcap"      , "RecoMuonPtAssMuonTrackIsoEndcap      " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr2         = new TH1D("RecoMuonPtAssMuonTrackIsoDr2"         , "RecoMuonPtAssMuonTrackIsoDr2         " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr2Barrel   = new TH1D("RecoMuonPtAssMuonTrackIsoDr2Barrel"   , "RecoMuonPtAssMuonTrackIsoDr2Barrel   " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr2Endcap   = new TH1D("RecoMuonPtAssMuonTrackIsoDr2Endcap"   , "RecoMuonPtAssMuonTrackIsoDr2Endcap   " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr02        = new TH1D("RecoMuonPtAssMuonTrackIsoDr02"        , "RecoMuonPtAssMuonTrackIsoDr02        " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr02Barrel  = new TH1D("RecoMuonPtAssMuonTrackIsoDr02Barrel"  , "RecoMuonPtAssMuonTrackIsoDr02Barrel  " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr02Endcap  = new TH1D("RecoMuonPtAssMuonTrackIsoDr02Endcap"  , "RecoMuonPtAssMuonTrackIsoDr02Endcap  " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr002       = new TH1D("RecoMuonPtAssMuonTrackIsoDr002"       , "RecoMuonPtAssMuonTrackIsoDr002       " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr002Barrel = new TH1D("RecoMuonPtAssMuonTrackIsoDr002Barrel" , "RecoMuonPtAssMuonTrackIsoDr002Barrel " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackIsoDr002Endcap = new TH1D("RecoMuonPtAssMuonTrackIsoDr002Endcap" , "RecoMuonPtAssMuonTrackIsoDr002Endcap " , 50 ,  0   , 50  );
   				    		 				    					    
  hRecoMuonPtAssMuonTrackNonIso            = new TH1D("RecoMuonPtAssMuonTrackNonIso"            , "RecoMuonPtAssMuonTrackNonIso            " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoBarrel      = new TH1D("RecoMuonPtAssMuonTrackNonIsoBarrel"      , "RecoMuonPtAssMuonTrackNonIsoBarrel      " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoEndcap      = new TH1D("RecoMuonPtAssMuonTrackNonIsoEndcap"      , "RecoMuonPtAssMuonTrackNonIsoEndcap      " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr2         = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr2"         , "RecoMuonPtAssMuonTrackNonIsoDr2         " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr2Barrel   = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr2Barrel"   , "RecoMuonPtAssMuonTrackNonIsoDr2Barrel   " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr2Endcap   = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr2Endcap"   , "RecoMuonPtAssMuonTrackNonIsoDr2Endcap   " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr02        = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr02"        , "RecoMuonPtAssMuonTrackNonIsoDr02        " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr02Barrel  = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr02Barrel"  , "RecoMuonPtAssMuonTrackNonIsoDr02Barrel  " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr02Endcap  = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr02Endcap"  , "RecoMuonPtAssMuonTrackNonIsoDr02Endcap  " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr002       = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr002"       , "RecoMuonPtAssMuonTrackNonIsoDr002       " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr002Barrel = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr002Barrel" , "RecoMuonPtAssMuonTrackNonIsoDr002Barrel " , 50 ,  0   , 50  );
  hRecoMuonPtAssMuonTrackNonIsoDr002Endcap = new TH1D("RecoMuonPtAssMuonTrackNonIsoDr002Endcap" , "RecoMuonPtAssMuonTrackNonIsoDr002Endcap " , 50 ,  0   , 50  );

   				    		 				    					    
  hRecoMuonEtaAssMuonTrackIso             = new TH1D("RecoMuonEtaAssMuonTrackIso"             , "RecoMuonEtaAssMuonTrackIso           " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoPt10 	  = new TH1D("RecoMuonEtaAssMuonTrackIsoPt10" 	      , "RecoMuonEtaAssMuonTrackIsoPt10       " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoPt20 	  = new TH1D("RecoMuonEtaAssMuonTrackIsoPt20" 	      , "RecoMuonEtaAssMuonTrackIsoPt20       " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr2          = new TH1D("RecoMuonEtaAssMuonTrackIsoDr2"          , "RecoMuonEtaAssMuonTrackIsoDr2          " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr2Pt10      = new TH1D("RecoMuonEtaAssMuonTrackIsoDr2Pt10"      , "RecoMuonEtaAssMuonTrackIsoDr2Pt10      " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr2Pt20      = new TH1D("RecoMuonEtaAssMuonTrackIsoDr2Pt20"      , "RecoMuonEtaAssMuonTrackIsoDr2Pt20      " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr02         = new TH1D("RecoMuonEtaAssMuonTrackIsoDr02"         , "RecoMuonEtaAssMuonTrackIsoDr02         " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr02Pt10     = new TH1D("RecoMuonEtaAssMuonTrackIsoDr02Pt10"     , "RecoMuonEtaAssMuonTrackIsoDr02Pt10     " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr02Pt20     = new TH1D("RecoMuonEtaAssMuonTrackIsoDr02Pt20"     , "RecoMuonEtaAssMuonTrackIsoDr02Pt20     " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr002        = new TH1D("RecoMuonEtaAssMuonTrackIsoDr002"        , "RecoMuonEtaAssMuonTrackIsoDr002        " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr002Pt10    = new TH1D("RecoMuonEtaAssMuonTrackIsoDr002Pt10"    , "RecoMuonEtaAssMuonTrackIsoDr002Pt10    " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackIsoDr002Pt20    = new TH1D("RecoMuonEtaAssMuonTrackIsoDr002Pt20"    , "RecoMuonEtaAssMuonTrackIsoDr002Pt20    " , 50 , -3   , 3   );

  hRecoMuonEtaAssMuonTrackNonIso           = new TH1D("RecoMuonEtaAssMuonTrackNonIso"             , "RecoMuonEtaAssMuonTrackNonIso           " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoPt10 	   = new TH1D("RecoMuonEtaAssMuonTrackNonIsoPt10" 	  , "RecoMuonEtaAssMuonTrackNonIsoPt10 	   " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoPt20 	   = new TH1D("RecoMuonEtaAssMuonTrackNonIsoPt20" 	  , "RecoMuonEtaAssMuonTrackNonIsoPt20 	   " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr2          = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr2"        , "RecoMuonEtaAssMuonTrackNonIsoDr2          " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr2Pt10      = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr2Pt10" 	  , "RecoMuonEtaAssMuonTrackNonIsoDr2Pt10      " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr2Pt20      = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr2Pt20" 	  , "RecoMuonEtaAssMuonTrackNonIsoDr2Pt20      " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr02         = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr02"       , "RecoMuonEtaAssMuonTrackNonIsoDr02         " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr02Pt10     = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr02Pt10"   , "RecoMuonEtaAssMuonTrackNonIsoDr02Pt10     " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr02Pt20     = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr02Pt20"   , "RecoMuonEtaAssMuonTrackNonIsoDr02Pt20     " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr002        = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr002"      , "RecoMuonEtaAssMuonTrackNonIsoDr002        " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr002Pt10    = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr002Pt10"  , "RecoMuonEtaAssMuonTrackNonIsoDr002Pt10    " , 50 , -3   , 3   );
  hRecoMuonEtaAssMuonTrackNonIsoDr002Pt20    = new TH1D("RecoMuonEtaAssMuonTrackNonIsoDr002Pt20"  , "RecoMuonEtaAssMuonTrackNonIsoDr002Pt20    " , 50 , -3   , 3   );





  //book the gen Histos			      		   			      	   			      
  hGenMuonPt                  = new TH1D("GenMuonPt"                  , "GenMuonPt                  " , 50 ,  0   , 50  ); 
  hGenMuonEta                 = new TH1D("GenMuonEta"                 , "GenMuonEta                 " , 50 , -3   , 3   );
  hGenMuonPhi                 = new TH1D("GenMuonPhi"                 , "GenMuonPhi                 " , 50 , -3.5   , 3.5   );
  hGenMuonMult                 = new TH1D("GenMuonMult"                 , "GenMuonMult                 " , 10 , 0   , 10   );

  hGenMuonPtBarrel              = new TH1D("GenMuonPtBarrel"             , "GenMuonPtBarrel             " , 50 ,  0   , 50  );                   
  hGenMuonPtEndcap              = new TH1D("GenMuonPtEndcap"             , "GenMuonPtEndcap             " , 50 ,  0   , 50  );                   

  hGenMuonEtaPt10               = new TH1D("GenMuonEtaPt10"              , "GenMuonEtaPt10              " , 50 ,  -3   , 3  );                   
  hGenMuonEtaPt20               = new TH1D("GenMuonEtaPt20"              , "GenMuonEtaPt20              " , 50 ,  -3   , 3  );                   


  // book histos for gen muons associated to HLT objects for path HLT1MuonIso
  for(unsigned int i=0; i<m_hlt1MuonIsoSrc.size(); i++) {
    myHistoName  = "GenMuonPtAssHLT1MuonIso_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Pt of GenMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hGenMuonPtAssHLT1MuonIso.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "GenMuonPtAssHLT1MuonIsoBarrel_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Barrel Pt of GenMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hGenMuonPtAssHLT1MuonIsoBarrel.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "GenMuonPtAssHLT1MuonIsoEndcap_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Endcap Pt of GenMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hGenMuonPtAssHLT1MuonIsoEndcap.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "GenMuonEtaAssHLT1MuonIso_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Eta of GenMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hGenMuonEtaAssHLT1MuonIso.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));    
    myHistoName  = "GenMuonEtaAssHLT1MuonIsoPt10_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Pt>10 Eta of GenMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hGenMuonEtaAssHLT1MuonIsoPt10.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));
    myHistoName  = "GenMuonEtaAssHLT1MuonIsoPt20_" + m_hlt1MuonIsoSrc[i];
    myHistoTitle = "Pt>20 Eta of GenMuon Ass. To HLT1MuonIso " + m_hlt1MuonIsoSrc[i];
    hGenMuonEtaAssHLT1MuonIsoPt20.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));
  }

  // book histos for gen muons associated to HLT objects for path HLT1MuonNonIso
  for(unsigned int i=0; i<m_hlt1MuonNonIsoSrc.size(); i++) {
    myHistoName  = "GenMuonPtAssHLT1MuonNonIso_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Pt of GenMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hGenMuonPtAssHLT1MuonNonIso.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "GenMuonPtAssHLT1MuonNonIsoBarrel_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Barrel Pt of GenMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hGenMuonPtAssHLT1MuonNonIsoBarrel.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "GenMuonPtAssHLT1MuonNonIsoEndcap_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Endcap Pt of GenMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hGenMuonPtAssHLT1MuonNonIsoEndcap.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  0   , 50  ));
    myHistoName  = "GenMuonEtaAssHLT1MuonNonIso_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Eta of GenMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hGenMuonEtaAssHLT1MuonNonIso.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));    
    myHistoName  = "GenMuonEtaAssHLT1MuonNonIsoPt10_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Pt>10 Eta of GenMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hGenMuonEtaAssHLT1MuonNonIsoPt10.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));
    myHistoName  = "GenMuonEtaAssHLT1MuonNonIsoPt20_" + m_hlt1MuonNonIsoSrc[i];
    myHistoTitle = "Pt>20 Eta of GenMuon Ass. To HLT1MuonNonIso " + m_hlt1MuonNonIsoSrc[i];
    hGenMuonEtaAssHLT1MuonNonIsoPt20.push_back(new TH1D(myHistoName.c_str(), myHistoTitle.c_str() , 50 ,  -3   , 3  ));
  }

  // book histos for gen muons associated to muon tracks used for trigger
  hGenMuonPtAssMuonTrackIso            = new TH1D("GenMuonPtAssMuonTrackIso"            , "GenMuonPtAssMuonTrackIso            " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoBarrel      = new TH1D("GenMuonPtAssMuonTrackIsoBarrel"      , "GenMuonPtAssMuonTrackIsoBarrel      " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoEndcap      = new TH1D("GenMuonPtAssMuonTrackIsoEndcap"      , "GenMuonPtAssMuonTrackIsoEndcap      " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr2         = new TH1D("GenMuonPtAssMuonTrackIsoDr2"         , "GenMuonPtAssMuonTrackIsoDr2         " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr2Barrel   = new TH1D("GenMuonPtAssMuonTrackIsoDr2Barrel"   , "GenMuonPtAssMuonTrackIsoDr2Barrel   " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr2Endcap   = new TH1D("GenMuonPtAssMuonTrackIsoDr2Endcap"   , "GenMuonPtAssMuonTrackIsoDr2Endcap   " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr02        = new TH1D("GenMuonPtAssMuonTrackIsoDr02"        , "GenMuonPtAssMuonTrackIsoDr02        " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr02Barrel  = new TH1D("GenMuonPtAssMuonTrackIsoDr02Barrel"  , "GenMuonPtAssMuonTrackIsoDr02Barrel  " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr02Endcap  = new TH1D("GenMuonPtAssMuonTrackIsoDr02Endcap"  , "GenMuonPtAssMuonTrackIsoDr02Endcap  " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr002       = new TH1D("GenMuonPtAssMuonTrackIsoDr002"       , "GenMuonPtAssMuonTrackIsoDr002       " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr002Barrel = new TH1D("GenMuonPtAssMuonTrackIsoDr002Barrel" , "GenMuonPtAssMuonTrackIsoDr002Barrel " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackIsoDr002Endcap = new TH1D("GenMuonPtAssMuonTrackIsoDr002Endcap" , "GenMuonPtAssMuonTrackIsoDr002Endcap " , 50 ,  0   , 50  );
   				    		 				    					    
  hGenMuonPtAssMuonTrackNonIso            = new TH1D("GenMuonPtAssMuonTrackNonIso"            , "GenMuonPtAssMuonTrackNonIso            " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoBarrel      = new TH1D("GenMuonPtAssMuonTrackNonIsoBarrel"      , "GenMuonPtAssMuonTrackNonIsoBarrel      " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoEndcap      = new TH1D("GenMuonPtAssMuonTrackNonIsoEndcap"      , "GenMuonPtAssMuonTrackNonIsoEndcap      " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr2         = new TH1D("GenMuonPtAssMuonTrackNonIsoDr2"         , "GenMuonPtAssMuonTrackNonIsoDr2         " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr2Barrel   = new TH1D("GenMuonPtAssMuonTrackNonIsoDr2Barrel"   , "GenMuonPtAssMuonTrackNonIsoDr2Barrel   " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr2Endcap   = new TH1D("GenMuonPtAssMuonTrackNonIsoDr2Endcap"   , "GenMuonPtAssMuonTrackNonIsoDr2Endcap   " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr02        = new TH1D("GenMuonPtAssMuonTrackNonIsoDr02"        , "GenMuonPtAssMuonTrackNonIsoDr02        " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr02Barrel  = new TH1D("GenMuonPtAssMuonTrackNonIsoDr02Barrel"  , "GenMuonPtAssMuonTrackNonIsoDr02Barrel  " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr02Endcap  = new TH1D("GenMuonPtAssMuonTrackNonIsoDr02Endcap"  , "GenMuonPtAssMuonTrackNonIsoDr02Endcap  " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr002       = new TH1D("GenMuonPtAssMuonTrackNonIsoDr002"       , "GenMuonPtAssMuonTrackNonIsoDr002       " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr002Barrel = new TH1D("GenMuonPtAssMuonTrackNonIsoDr002Barrel" , "GenMuonPtAssMuonTrackNonIsoDr002Barrel " , 50 ,  0   , 50  );
  hGenMuonPtAssMuonTrackNonIsoDr002Endcap = new TH1D("GenMuonPtAssMuonTrackNonIsoDr002Endcap" , "GenMuonPtAssMuonTrackNonIsoDr002Endcap " , 50 ,  0   , 50  );
   				    		 				    					    
   				    		 				    					    
  hGenMuonEtaAssMuonTrackIso           = new TH1D("GenMuonEtaAssMuonTrackIso"               , "GenMuonEtaAssMuonTrackIso           " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoPt10       = new TH1D("GenMuonEtaAssMuonTrackIsoPt10" 	    , "GenMuonEtaAssMuonTrackIsoPt10       " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoPt20       = new TH1D("GenMuonEtaAssMuonTrackIsoPt20" 	    , "GenMuonEtaAssMuonTrackIsoPt20       " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr2          = new TH1D("GenMuonEtaAssMuonTrackIsoDr2"          , "GenMuonEtaAssMuonTrackIsoDr2          " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr2Pt10      = new TH1D("GenMuonEtaAssMuonTrackIsoDr2Pt10" 	    , "GenMuonEtaAssMuonTrackIsoDr2Pt10      " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr2Pt20      = new TH1D("GenMuonEtaAssMuonTrackIsoDr2Pt20" 	    , "GenMuonEtaAssMuonTrackIsoDr2Pt20      " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr02         = new TH1D("GenMuonEtaAssMuonTrackIsoDr02"         , "GenMuonEtaAssMuonTrackIsoDr02         " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr02Pt10     = new TH1D("GenMuonEtaAssMuonTrackIsoDr02Pt10"     , "GenMuonEtaAssMuonTrackIsoDr02Pt10     " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr02Pt20     = new TH1D("GenMuonEtaAssMuonTrackIsoDr02Pt20"     , "GenMuonEtaAssMuonTrackIsoDr02Pt20     " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr002        = new TH1D("GenMuonEtaAssMuonTrackIsoDr002"        , "GenMuonEtaAssMuonTrackIsoDr002        " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr002Pt10    = new TH1D("GenMuonEtaAssMuonTrackIsoDr002Pt10"    , "GenMuonEtaAssMuonTrackIsoDr002Pt10    " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackIsoDr002Pt20    = new TH1D("GenMuonEtaAssMuonTrackIsoDr002Pt20"    , "GenMuonEtaAssMuonTrackIsoDr002Pt20    " , 50 , -3   , 3   );

  hGenMuonEtaAssMuonTrackNonIso           = new TH1D("GenMuonEtaAssMuonTrackNonIso"           	  , "GenMuonEtaAssMuonTrackNonIso           " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoPt10       = new TH1D("GenMuonEtaAssMuonTrackNonIsoPt10"       	  , "GenMuonEtaAssMuonTrackNonIsoPt10 	" , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoPt20       = new TH1D("GenMuonEtaAssMuonTrackNonIsoPt20"       	  , "GenMuonEtaAssMuonTrackNonIsoPt20 	" , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr2          = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr2"          , "GenMuonEtaAssMuonTrackNonIsoDr2          " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr2Pt10      = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr2Pt10"      , "GenMuonEtaAssMuonTrackNonIsoDr2Pt10      " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr2Pt20      = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr2Pt20"      , "GenMuonEtaAssMuonTrackNonIsoDr2Pt20      " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr02         = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr02"         , "GenMuonEtaAssMuonTrackNonIsoDr02         " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr02Pt10     = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr02Pt10"     , "GenMuonEtaAssMuonTrackNonIsoDr02Pt10     " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr02Pt20     = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr02Pt20"     , "GenMuonEtaAssMuonTrackNonIsoDr02Pt20     " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr002        = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr002"        , "GenMuonEtaAssMuonTrackNonIsoDr002        " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr002Pt10    = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr002Pt10"    , "GenMuonEtaAssMuonTrackNonIsoDr002Pt10    " , 50 , -3   , 3   );
  hGenMuonEtaAssMuonTrackNonIsoDr002Pt20    = new TH1D("GenMuonEtaAssMuonTrackNonIsoDr002Pt20"    , "GenMuonEtaAssMuonTrackNonIsoDr002Pt20    " , 50 , -3   , 3   );

}



void TurnOnMaker::handleObjects(const edm::Event& iEvent)
{



  //*******************************************************
  // Get the HLT Objects through the TriggerEventWithRefs
  //*******************************************************


  // Get the Trigger collection
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  iEvent.getByLabel("triggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("HLTSusyBSMVal") << "RAW-type HLT results not found, skipping event";
    return;
  }


  //clear the vectors 
  for(unsigned int i=0; i<theHLT1MuonIsoObjectVector.size(); i++) {theHLT1MuonIsoObjectVector[i].clear();}
  for(unsigned int i=0; i<theHLT1MuonNonIsoObjectVector.size(); i++) {theHLT1MuonNonIsoObjectVector[i].clear();}





  //HLT1MuonIso
  for(unsigned int i=0; i<theHLT1MuonIsoObjectVector.size(); i++) {
    if ( triggerObj->filterIndex(m_hlt1MuonIsoSrc[i])>=triggerObj->size() ) {
      cout <<"No HLT Collection with label "<< m_hlt1MuonIsoSrc[i] << endl;
      //      LogDebug("HLTSusyBSMVal")<<"No HLT Collection with label "<< m_hlt1MuonIsoSrc[i];
      break ;
    }
    triggerObj->getObjects(triggerObj->filterIndex(m_hlt1MuonIsoSrc[i]),TriggerMuon, theHLT1MuonIsoObjectVector[i]);
  }


  //HLT1MuonNonIso
  for(unsigned int i=0; i<theHLT1MuonNonIsoObjectVector.size(); i++) {
    if ( triggerObj->filterIndex(m_hlt1MuonNonIsoSrc[i])>=triggerObj->size() ) {
      cout <<"No HLT Collection with label "<<m_hlt1MuonNonIsoSrc[i] << endl;
      //      LogDebug("HLTSusyBSMVal")<<"No HLT Collection with label "<<m_hlt1MuonNonIsoSrc[i] ;
      break ;
    }
    triggerObj->getObjects(triggerObj->filterIndex(m_hlt1MuonNonIsoSrc[i]),TriggerMuon, theHLT1MuonNonIsoObjectVector[i]);
  }

  //***********************************************
  // Get the HLT basic objects
  //***********************************************

 // Get the muon tracks
 // used to build the HLT muons
 // to build plots without the
 // hardcoded cuts in the trigger

   // get hold of trks
   Handle<RecoChargedCandidateCollection> theMuonTrackCandidateHandle;
   theMuonTrackCollection.clear();
   try {
     iEvent.getByLabel (m_hltMuonTrackSrc, theMuonTrackCandidateHandle);
     theMuonTrackCollection = *theMuonTrackCandidateHandle;
     std::sort(theMuonTrackCollection.begin(), theMuonTrackCollection.end(), PtSorter());
   } catch (...) {
     cout << "No RecoChargedCandidateCollection with label " << m_hltMuonTrackSrc << endl;
   }





  //***********************************************
  // Get the RECO Objects
  //***********************************************

  //Get the Muons
  Handle<MuonCollection> theMuonCollectionHandle; 
  iEvent.getByLabel(m_recoMuonSrc, theMuonCollectionHandle);
  theMuonCollection = *theMuonCollectionHandle;
  std::sort(theMuonCollection.begin(), theMuonCollection.end(), PtSorter());

  //***********************************************
  // Get the MC truth Objects
  //***********************************************
  Handle<reco::CandidateCollection> theCandidateCollectionHandle;
  iEvent.getByLabel(m_genSrc, theCandidateCollectionHandle);
  theGenParticleCollection = theCandidateCollectionHandle.product();

}



// bool TurnOnMaker::recoToTriggerMatched(reco::Candidate* theRecoObject, std::vector< edm::RefToBase< reco::Candidate > > theTriggerObjectVector) {
//   bool decision = false;
//   for(unsigned int i=0; i<theTriggerObjectVector.size(); i++) {
//     double Deta = theRecoObject->eta() - theTriggerObjectVector[i]->eta();
//     double Dphi = theRecoObject->phi() - theTriggerObjectVector[i]->phi();
//     double DR = sqrt(Deta*Deta+Dphi*Dphi);
//     //    if(DR<0.5) {
//     if(DR<1) {
//       decision = true;
//       break;
//     }
//   }
//   return decision;
// }

// bool TurnOnMaker::recoToTriggerMatched(const reco::Candidate* theRecoObject, std::vector< edm::RefToBase< reco::Candidate > > theTriggerObjectVector) {
//   bool decision = false;
//   for(unsigned int i=0; i<theTriggerObjectVector.size(); i++) {
//     double Deta = theRecoObject->eta() - theTriggerObjectVector[i]->eta();
//     double Dphi = theRecoObject->phi() - theTriggerObjectVector[i]->phi();
//     double DR = sqrt(Deta*Deta+Dphi*Dphi);
//     //    if(DR<0.5) {
//     if(DR<1) {
//       decision = true;
//       break;
//     }
//   }
//   return decision;
// }


//function for the association of Reco objects with HLT objects
bool TurnOnMaker::recoToTriggerMatched(reco::Candidate* theRecoObject, std::vector< std::vector<RecoChargedCandidateRef> > theTriggerLevelObjectVector, int iHltLevel) {
  bool decision = false;
  if(iHltLevel==0) {
    for(unsigned int i=0; i<theTriggerLevelObjectVector[0].size(); i++) {
      double Deta = theRecoObject->eta() - theTriggerLevelObjectVector[0][i]->eta();
      double Dphi = theRecoObject->phi() - theTriggerLevelObjectVector[0][i]->phi();
      double DR = sqrt(Deta*Deta+Dphi*Dphi);
      if(DR<0.5) {
	decision = true;
	break;
      }
    }
  }
  if(iHltLevel>0) {
    for(unsigned int i=0; i<theTriggerLevelObjectVector[iHltLevel].size(); i++) {
      if(triggerToTriggerMatched(theTriggerLevelObjectVector[iHltLevel][i],theTriggerLevelObjectVector,iHltLevel-1)) {
	double Deta = theRecoObject->eta() - theTriggerLevelObjectVector[iHltLevel][i]->eta();
	double Dphi = theRecoObject->phi() - theTriggerLevelObjectVector[iHltLevel][i]->phi();
	double DR = sqrt(Deta*Deta+Dphi*Dphi);
	if(DR<0.5) {
	  decision = true;
	  break;
	}
      }
    }
  }
  return decision;
}

//function for the association of Gen objects with HLT objects
bool TurnOnMaker::recoToTriggerMatched(const reco::Candidate* theRecoObject, std::vector< std::vector<RecoChargedCandidateRef> > theTriggerLevelObjectVector, int iHltLevel) {
  bool decision = false;
  if(iHltLevel==0) {
    for(unsigned int i=0; i<theTriggerLevelObjectVector[0].size(); i++) {
      double Deta = theRecoObject->eta() - theTriggerLevelObjectVector[0][i]->eta();
      double Dphi = theRecoObject->phi() - theTriggerLevelObjectVector[0][i]->phi();
      double DR = sqrt(Deta*Deta+Dphi*Dphi);
      if(DR<0.5) {
	decision = true;
	break;
      }
    }
  }
  if(iHltLevel>0) {
    for(unsigned int i=0; i<theTriggerLevelObjectVector[iHltLevel].size(); i++) {
      if(triggerToTriggerMatched(theTriggerLevelObjectVector[iHltLevel][i],theTriggerLevelObjectVector,iHltLevel-1)) {
	double Deta = theRecoObject->eta() - theTriggerLevelObjectVector[iHltLevel][i]->eta();
	double Dphi = theRecoObject->phi() - theTriggerLevelObjectVector[iHltLevel][i]->phi();
	double DR = sqrt(Deta*Deta+Dphi*Dphi);
	if(DR<0.5) {
	  decision = true;
	  break;
	}
      }
    }
  }
  return decision;
}


//function for the association of Reco objects with muon tracks used by the trigger
bool TurnOnMaker::recoToTracksMatched(reco::Candidate* theRecoObject, reco::RecoChargedCandidateCollection theRecoChargedCandidateCollection, double Dr, string Cond) {
  bool decision = false;
  double pt_min;

  if(Cond == s_Iso)         pt_min = 11;
  else if(Cond == s_NonIso) pt_min = 16;
  else return decision;
  
  
  for(unsigned int i=0; i<theRecoChargedCandidateCollection.size(); i++) {
    TrackRef tk = theRecoChargedCandidateCollection[i].get<TrackRef>();
    if (fabs(tk->eta())>2.5) continue;
    //    if (tk->numberOfValidHits()<0) continue;
    if (fabs(tk->d0())>Dr) continue;
    if (fabs(tk->dz())>9999) continue;
    double pt = tk->pt();
    double err0 = tk->error(0);
    double abspar0 = fabs(tk->parameter(0));
    double ptLx = pt;
    // convert 50% efficiency threshold to 90% efficiency threshold
    if (abspar0>0) ptLx += 2.2*err0/abspar0*pt;
    if (ptLx<pt_min) continue;
    
    double Deta = theRecoObject->eta() - tk->eta();
    double Dphi = theRecoObject->phi() - tk->phi();
    double DR = sqrt(Deta*Deta+Dphi*Dphi);
    if(DR<0.5) {
      decision = true;
      break;
    }
  }
  return decision;
}



//function for the association of Gen objects with muon tracks used by the trigger
bool TurnOnMaker::recoToTracksMatched(const reco::Candidate* theRecoObject, reco::RecoChargedCandidateCollection theRecoChargedCandidateCollection, double Dr, string Cond) {
  bool decision = false;
  double pt_min;

  if(Cond == s_Iso)         pt_min = 11;
  else if(Cond == s_NonIso) pt_min = 16;
  else return decision;
  
  
  for(unsigned int i=0; i<theRecoChargedCandidateCollection.size(); i++) {
    TrackRef tk = theRecoChargedCandidateCollection[i].get<TrackRef>();
    if (fabs(tk->eta())>2.5) continue;
    //    if (tk->numberOfValidHits()<0) continue;
    if (fabs(tk->d0())>Dr) continue;
    if (fabs(tk->dz())>9999) continue;
    double pt = tk->pt();
    double err0 = tk->error(0);
    double abspar0 = fabs(tk->parameter(0));
    double ptLx = pt;
    // convert 50% efficiency threshold to 90% efficiency threshold
    if (abspar0>0) ptLx += 2.2*err0/abspar0*pt;
    if (ptLx<pt_min) continue;
    
    double Deta = theRecoObject->eta() - tk->eta();
    double Dphi = theRecoObject->phi() - tk->phi();
    double DR = sqrt(Deta*Deta+Dphi*Dphi);
    if(DR<0.5) {
      decision = true;
      break;
    }
  }
  return decision;
}





//function for the association of HLT objects with HLT objects
bool TurnOnMaker::triggerToTriggerMatched(RecoChargedCandidateRef theTriggerObject, std::vector< std::vector<RecoChargedCandidateRef> > theTriggerLevelObjectVector, int iHltLevel) {
  bool decision = false;
  if(iHltLevel==0) {
    for(unsigned int i=0; i<theTriggerLevelObjectVector[0].size(); i++) {
      double Deta = theTriggerObject->eta() - theTriggerLevelObjectVector[0][i]->eta();
      double Dphi = theTriggerObject->phi() - theTriggerLevelObjectVector[0][i]->phi();
      double DR = sqrt(Deta*Deta+Dphi*Dphi);
      if(DR<0.5) {
	decision = true;
	break;
      }
    }
  }
  if(iHltLevel>0) {
    for(unsigned int i=0; i<theTriggerLevelObjectVector[iHltLevel].size(); i++) {
      if(triggerToTriggerMatched(theTriggerLevelObjectVector[iHltLevel][i],theTriggerLevelObjectVector,iHltLevel-1)) {
	double Deta = theTriggerObject->eta() - theTriggerLevelObjectVector[iHltLevel][i]->eta();
	double Dphi = theTriggerObject->phi() - theTriggerLevelObjectVector[iHltLevel][i]->phi();
	double DR = sqrt(Deta*Deta+Dphi*Dphi);
	if(DR<0.5) {
	  decision = true;
	  break;
	}
      }
    }	
  }
  return decision;
}
     
  


