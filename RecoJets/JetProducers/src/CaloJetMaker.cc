#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "PhysicsTools/RecoCandidate/interface/RecoCandidate.h"

#include "RecoJets/JetProducers/interface/CaloJetMaker.h"

using namespace std;
using namespace reco;

CaloJet CaloJetMaker::makeCaloJet (const ProtoJet2& fProtojet) const {
  std::vector<CaloTowerDetId> towerIds;
  const CaloTowerCollection* towerCollection = 0;
  const ProtoJet2::Candidates* towers = &fProtojet.getTowerList();
  towerIds.reserve (towers->size ());
  ProtoJet2::Candidates::const_iterator tower = towers->begin ();
  for (; tower != towers->end (); tower++) {
    edm::Ref<CaloTowerCollection> towerRef = component<CaloTower>::get (**tower);
    if (towerRef.isNonnull ()) { // valid
      const CaloTowerCollection* newproduct = towerRef.product ();
      if (!towerCollection) towerCollection  = newproduct;
      else if (towerCollection != newproduct) {
	cerr << "CaloJetMaker::makeCaloJet (const ProtoJet2& fProtojet) ERROR-> "
	     << "CaloTower collection for tower is not the same. Previous: " <<  towerCollection 
	     << ", new: " << newproduct << endl;
      }
      towerIds.push_back (towerRef->id ());
    }
    else {
      cerr << "CaloJetMaker::makeCaloJet (const ProtoJet2& fProtojet) ERROR-> "
	   << "invalid reco::CaloTowerRef towerRef = tower->caloTower()" << endl;
    }
  }
  return CaloJet (fProtojet.px(), fProtojet.py(), fProtojet.pz(), 
		  fProtojet.e(), fProtojet.p(), fProtojet.pt(), fProtojet.et(), fProtojet.m(), 
		  fProtojet.phi(), fProtojet.eta(), fProtojet.y(), fProtojet.numberOfConstituents(), 
		  *towerCollection, towerIds);
}

