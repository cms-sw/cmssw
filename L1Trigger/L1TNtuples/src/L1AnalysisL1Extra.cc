#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1Extra.h"

L1Analysis::L1AnalysisL1Extra::L1AnalysisL1Extra()
{
}

L1Analysis::L1AnalysisL1Extra::~L1AnalysisL1Extra()
{

}

void L1Analysis::L1AnalysisL1Extra::SetIsoEm(const edm::Handle<l1extra::L1EmParticleCollection> isoEm, unsigned maxL1Extra)
{
  for(l1extra::L1EmParticleCollection::const_iterator it=isoEm->begin(); it!=isoEm->end() && l1extra_.nIsoEm<maxL1Extra; it++){
      
      l1extra_.isoEmEt .push_back(it->et());
      l1extra_.isoEmEta.push_back(it->eta());
      l1extra_.isoEmPhi.push_back(it->phi());
      l1extra_.isoEmBx .push_back(it->bx());
      l1extra_.nIsoEm++;
    }
}


void L1Analysis::L1AnalysisL1Extra::SetNonIsoEm(const edm::Handle<l1extra::L1EmParticleCollection> nonIsoEm, unsigned maxL1Extra)
{
  for(l1extra::L1EmParticleCollection::const_iterator it=nonIsoEm->begin(); it!=nonIsoEm->end() && l1extra_.nNonIsoEm<maxL1Extra; it++){
      
      l1extra_.nonIsoEmEt .push_back(it->et());
      l1extra_.nonIsoEmEta.push_back(it->eta());
      l1extra_.nonIsoEmPhi.push_back(it->phi());
      l1extra_.nonIsoEmBx .push_back(it->bx());
      l1extra_.nNonIsoEm++;
    }
}

void L1Analysis::L1AnalysisL1Extra::SetCenJet(const edm::Handle<l1extra::L1JetParticleCollection> cenJet, unsigned maxL1Extra)
{
//      std::cout << "Filling L1 Extra cenJets" << maxL1Extra << " " << cenJet->size() << std::endl;      
 
  for(l1extra::L1JetParticleCollection::const_iterator it=cenJet->begin(); it!=cenJet->end() && l1extra_.nCenJets<maxL1Extra; it++){
      //printf("L1CenJet (et,eta,phi,bx,) (%f,%f,%f,%d) \n",it->et(),it->eta(),it->phi(),it->bx() );
//      std::cout << "L1 CenJets et,eta,phi,bx = " << it->et() << ", " << it->eta() <<", " <<it->phi() <<", " << it->bx() << std::endl;
      l1extra_.cenJetEt .push_back(it->et());
      l1extra_.cenJetEta.push_back(it->eta());
      l1extra_.cenJetPhi.push_back(it->phi());
      l1extra_.cenJetBx .push_back(it->bx());
      l1extra_.nCenJets++;
    }
}

void L1Analysis::L1AnalysisL1Extra::SetFwdJet(const edm::Handle<l1extra::L1JetParticleCollection> fwdJet, unsigned maxL1Extra)
{
      //std::cout << "Filling L1 Extra fwdJets" << std::endl;      
   for(l1extra::L1JetParticleCollection::const_iterator it=fwdJet->begin(); it!=fwdJet->end() && l1extra_.nFwdJets<maxL1Extra; it++){
      
      //printf("L1fwdJet (et,eta,phi,bx,) (%f,%f,%f,%d)\n",it->et(),it->eta(),it->phi(),it->bx() );
      l1extra_.fwdJetEt .push_back(it->et());
      l1extra_.fwdJetEta.push_back(it->eta());
      l1extra_.fwdJetPhi.push_back(it->phi());
      l1extra_.fwdJetBx .push_back(it->bx());
      l1extra_.nFwdJets++;
    }
}

void L1Analysis::L1AnalysisL1Extra::SetTauJet(const edm::Handle<l1extra::L1JetParticleCollection> tauJet, unsigned maxL1Extra)
{
      //std::cout << "Filling L1 Extra tauJets" << std::endl;      
   for(l1extra::L1JetParticleCollection::const_iterator it=tauJet->begin(); it!=tauJet->end() && l1extra_.nTauJets<maxL1Extra; it++){
      
     // printf("L1tauJet (et,eta,phi,bx,) (%f,%f,%f,%d)\n",it->et(),it->eta(),it->phi(),it->bx() );
      l1extra_.tauJetEt .push_back(it->et());
      l1extra_.tauJetEta.push_back(it->eta());
      l1extra_.tauJetPhi.push_back(it->phi());
      l1extra_.tauJetBx .push_back(it->bx());
      l1extra_.nTauJets++;
    }
}

void L1Analysis::L1AnalysisL1Extra::SetIsoTauJet(const edm::Handle<l1extra::L1JetParticleCollection> isoTauJet, unsigned maxL1Extra)
{
  // std::cout << "Filling L1 Extra isoTauJets" << std::endl;      
   for(l1extra::L1JetParticleCollection::const_iterator it=isoTauJet->begin(); it!=isoTauJet->end() && l1extra_.nIsoTauJets<maxL1Extra; it++){
      
     // printf("L1isoTauJet (et,eta,phi,bx,) (%f,%f,%f,%d)\n",it->et(),it->eta(),it->phi(),it->bx() );
      l1extra_.isoTauJetEt .push_back(it->et());
      l1extra_.isoTauJetEta.push_back(it->eta());
      l1extra_.isoTauJetPhi.push_back(it->phi());
      l1extra_.isoTauJetBx .push_back(it->bx());
      l1extra_.nIsoTauJets++;
    }
}


void L1Analysis::L1AnalysisL1Extra::SetMuon(const edm::Handle<l1extra::L1MuonParticleCollection> muon, unsigned maxL1Extra)
{
  for(l1extra::L1MuonParticleCollection::const_iterator it=muon->begin(); it!=muon->end() && l1extra_.nMuons<maxL1Extra; it++){
      
      l1extra_.muonEt .push_back( it->et());
      l1extra_.muonEta.push_back(it->eta());
      l1extra_.muonPhi.push_back(it->phi());
      l1extra_.muonChg.push_back(it->charge());
      l1extra_.muonIso.push_back(it->isIsolated());
      l1extra_.muonMip.push_back(it->isMip());
      l1extra_.muonFwd.push_back(it->isForward());
      l1extra_.muonRPC.push_back(it->isRPC());
      l1extra_.muonBx .push_back(it->bx());
      l1extra_.muonQuality .push_back(it->gmtMuonCand().quality());
		
//		std::cout << "gmtmuon cand: pt " << it->gmtMuonCand().ptValue() 
//					<< "; ptExtra " << it->et() 
//					<< "; qual " << it->gmtMuonCand().quality() 
//					<< std::endl;
      l1extra_.nMuons++;
    }
}

void L1Analysis::L1AnalysisL1Extra::SetMet(const edm::Handle<l1extra::L1EtMissParticleCollection> mets)
{
  for(l1extra::L1EtMissParticleCollection::const_iterator it=mets->begin(); it!=mets->end(); it++) {
    l1extra_.et.    push_back( it->etTotal() ); 
    l1extra_.met.   push_back( it->et() );
    l1extra_.metPhi.push_back( it->phi() );
    l1extra_.metBx. push_back( it->bx() );
    l1extra_.nMet++;
  }
}

void L1Analysis::L1AnalysisL1Extra::SetMht(const edm::Handle<l1extra::L1EtMissParticleCollection> mhts)
{
  for(l1extra::L1EtMissParticleCollection::const_iterator it=mhts->begin(); it!=mhts->end(); it++) {
    l1extra_.ht.    push_back( it->etTotal() );
    l1extra_.mht.   push_back( it->et() );
    l1extra_.mhtPhi.push_back( it->phi() );
    l1extra_.mhtBx. push_back( it->bx() );
    l1extra_.nMht++;
  }
}

void L1Analysis::L1AnalysisL1Extra::SetHFring(const edm::Handle<l1extra::L1HFRingsCollection> hfRings)
{  
   l1extra_.hfEtSum.resize(4);
   l1extra_.hfBitCnt.resize(4);
   l1extra_.hfBx.resize(4);
  
    for(unsigned int i=0; i<4; ++i)  
    {
      if (hfRings->size()==0) continue;

      l1extra_.hfEtSum[i] = (hfRings->begin()->hfEtSum((l1extra::L1HFRings::HFRingLabels) i));
      l1extra_.hfBitCnt[i] = (hfRings->begin()->hfBitCount((l1extra::L1HFRings::HFRingLabels) i));
      l1extra_.hfBx[i] = hfRings->begin()->bx();
      }
}


