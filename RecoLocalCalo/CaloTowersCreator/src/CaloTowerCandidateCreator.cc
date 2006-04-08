// makes CaloTowerCandidates from CaloTowers
// original author: L.Lista INFN
// modifyed by: F.Ratnikov UMd
// $Id: CaloTowerCandidateCreator.cc,v 1.6 2006/04/04 06:47:36 llista Exp $
#include <cmath>
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowerCandidateCreator.h"
using namespace edm;
using namespace reco;
using namespace std;

CaloTowerCandidateCreator::CaloTowerCandidateCreator( const ParameterSet & p ) :
  source( p.getParameter<string>( "src" ) ) {
  produces<CandidateCollection>();
}

CaloTowerCandidateCreator::~CaloTowerCandidateCreator() {
}

void CaloTowerCandidateCreator::produce( Event& evt, const EventSetup& ) {
  Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel( source, caloTowers );
  
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( caloTowers->size() );
  unsigned idx = 0;
  for (; idx < caloTowers->size (); idx++) {
    const CaloTower* cal = &((*caloTowers) [idx]);
    math::PtEtaPhiELorentzVector p( cal->et(), cal->eta(), cal->phi(), cal->energy() );
    RecoCaloTowerCandidate * c = 
      new RecoCaloTowerCandidate( 0, Candidate::LorentzVector( p ) );
    c->setCaloTower( RecoCandidate::CaloTowerRef( caloTowers, idx) );
    cands->push_back( c );
  }
  evt.put( cands );
}
