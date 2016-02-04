// makes CaloTowerCandidates from CaloTowers
// original author: L.Lista INFN
// modifyed by: F.Ratnikov UMd
// $Id: CaloTowerCandidateCreator.cc,v 1.10 2008/03/04 04:47:26 anastass Exp $
#include <cmath>
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowerCandidateCreator.h"
using namespace edm;
using namespace reco;
using namespace std;

CaloTowerCandidateCreator::CaloTowerCandidateCreator( const ParameterSet & p ) 
  :
  mVerbose (p.getUntrackedParameter<int> ("verbose", 0)),
  mSource (p.getParameter<edm::InputTag> ("src")),
  mEtThreshold (p.getParameter<double> ("minimumEt")),
  mEThreshold (p.getParameter<double> ("minimumE"))
{
  produces<CandidateCollection>();
}

CaloTowerCandidateCreator::~CaloTowerCandidateCreator() {
}

void CaloTowerCandidateCreator::produce( Event& evt, const EventSetup& ) {
  Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel( mSource, caloTowers );
  
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( caloTowers->size() );
  unsigned idx = 0;
  for (; idx < caloTowers->size (); idx++) {
    const CaloTower* cal = &((*caloTowers) [idx]);
    if (mVerbose >= 2) {
      std::cout << "CaloTowerCandidateCreator::produce-> " << idx << " tower et/eta/phi/e: " 
		<< cal->et() << '/' << cal->eta() << '/' << cal->phi() << '/' << cal->energy() << " is...";
    }
    if (cal->et() >= mEtThreshold && cal->energy() >= mEThreshold ) {
      math::PtEtaPhiMLorentzVector p( cal->et(), cal->eta(), cal->phi(), 0 );
      RecoCaloTowerCandidate * c = 
	new RecoCaloTowerCandidate( 0, Candidate::LorentzVector( p ) );
      c->setCaloTower (CaloTowerRef( caloTowers, idx) );
      cands->push_back( c );
      if (mVerbose >= 2) std::cout << "accepted: pT/eta/phi:" << c->pt() << '/' << c->eta() <<  '/' << c->phi() <<std::endl;
    }
    else {
      if (mVerbose >= 2) std::cout << "rejected" << std::endl;
    }
  }
  if (mVerbose >= 1) {
    std::cout << "CaloTowerCandidateCreator::produce-> " << cands->size () << " candidates created" << std::endl;
  }
  evt.put( cands );
}
