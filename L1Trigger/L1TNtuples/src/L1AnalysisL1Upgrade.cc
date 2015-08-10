#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1Upgrade.h"

L1Analysis::L1AnalysisL1Upgrade::L1AnalysisL1Upgrade()
{
}

L1Analysis::L1AnalysisL1Upgrade::~L1AnalysisL1Upgrade()
{

}

void L1Analysis::L1AnalysisL1Upgrade::SetEm(const edm::Handle<l1t::EGammaBxCollection> em, unsigned maxL1Upgrade)
{
  for (l1t::EGammaBxCollection::const_iterator it=em->begin(0); it!=em->end(0) && l1upgrade_.nEGs<maxL1Upgrade; it++){
      l1upgrade_.egEt .push_back(it->pt());
      l1upgrade_.egEta.push_back(it->eta());
      l1upgrade_.egPhi.push_back(it->phi());
      l1upgrade_.egIso.push_back(it->hwIso());
      l1upgrade_.egBx .push_back(0);
      l1upgrade_.nEGs++;
    }
}


void L1Analysis::L1AnalysisL1Upgrade::SetTau(const edm::Handle<l1t::TauBxCollection> tau, unsigned maxL1Upgrade)
{
  for (l1t::TauBxCollection::const_iterator it=tau->begin(0); it!=tau->end(0) && l1upgrade_.nTaus<maxL1Upgrade; it++){
      l1upgrade_.tauEt .push_back(it->et());
      l1upgrade_.tauEta.push_back(it->eta());
      l1upgrade_.tauPhi.push_back(it->phi());
      l1upgrade_.tauIso.push_back(it->hwIso());
      l1upgrade_.tauBx .push_back(0);
      l1upgrade_.nTaus++;
    }
}


void L1Analysis::L1AnalysisL1Upgrade::SetJet(const edm::Handle<l1t::JetBxCollection> jet, unsigned maxL1Upgrade)
{
  for (l1t::JetBxCollection::const_iterator it=jet->begin(0); it!=jet->end(0) && l1upgrade_.nJets<maxL1Upgrade; it++){
      l1upgrade_.jetEt .push_back(it->et());
      l1upgrade_.jetEta.push_back(it->eta());
      l1upgrade_.jetPhi.push_back(it->phi());
      l1upgrade_.jetBx .push_back(0);
      l1upgrade_.nJets++;
    }
}


void L1Analysis::L1AnalysisL1Upgrade::SetMuon(const edm::Handle<l1t::MuonBxCollection> muon, unsigned maxL1Upgrade)
{
  for (l1t::MuonBxCollection::const_iterator it=muon->begin(0); it!=muon->end(0) && l1upgrade_.nMuons<maxL1Upgrade; it++){
      
      l1upgrade_.muonEt .push_back(it->et());
      l1upgrade_.muonEta.push_back(it->eta());
      l1upgrade_.muonPhi.push_back(it->phi());
      l1upgrade_.muonChg.push_back(0); //it->charge());
      l1upgrade_.muonIso.push_back(it->hwIso());
      l1upgrade_.muonBx .push_back(0);
      l1upgrade_.nMuons++;
    }
}

void L1Analysis::L1AnalysisL1Upgrade::SetSum(const edm::Handle<l1t::EtSumBxCollection> sums, unsigned maxL1Upgrade)
{
  for (l1t::EtSumBxCollection::const_iterator it=sums->begin(0); it!=sums->end(0) && l1upgrade_.nSums<maxL1Upgrade; it++) {
    l1upgrade_.sumEt. push_back( it->et() ); 
    l1upgrade_.sumPhi.push_back( it->phi() );
    l1upgrade_.sumBx. push_back( 0 );
    l1upgrade_.nSums++;
  }
}


