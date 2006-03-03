#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "PhysicsTools/RecoCandidate/interface/RecoCandidate.h"

#include "RecoJets/JetAlgorithms/interface/CaloJetMaker.h"

using namespace std;
using namespace reco;

namespace {
  bool makeSpecific (const CaloTowerCollection& fTowers,
		     const std::vector<CaloTowerDetId>& fTowerIds,
		     CaloJet::Specific* fJetSpecific) {
    if (!fJetSpecific) return false;
    
    // 1.- Loop over the tower Ids, 
    // 2.- Get the corresponding CaloTower
    // 3.- Calculate the different CaloJet specific quantities
    vector<double> eECal_i;
    vector<double> eHCal_i;
    double eInHad = 0.;
    double eInEm = 0.;
    double eInHO = 0.;
    double eInHB = 0.;
    double eInHF = 0.;
    double eInHE = 0.;
    
    for(vector<CaloTowerDetId>::const_iterator i = fTowerIds.begin(); i != fTowerIds.end(); ++i) {
      const CaloTower* aTower =  &*fTowers.find(*i);
      //Array of energy in EM Towers:
      eECal_i.push_back(aTower->e_em());
      eInEm += aTower->e_em();
      //Array of energy in HCAL Towers:
      eHCal_i.push_back(aTower->e_had()); 
      eInHad += aTower->e_had();
      
      eInHO += aTower->e_outer();
      // have no data for eInHB eInHE eInHF
    }
    double towerEnergy = eInHad + eInEm;
    fJetSpecific->m_energyFractionInHO = eInHO / towerEnergy;
    fJetSpecific->m_energyFractionInHB = eInHB / towerEnergy;
    fJetSpecific->m_energyFractionInHE = eInHE / towerEnergy;
    fJetSpecific->m_energyFractionInHF = eInHF / towerEnergy;
    fJetSpecific->m_energyFractionInHCAL = eInHad / towerEnergy;
    fJetSpecific->m_energyFractionInECAL = eInEm / towerEnergy;
    
    //Sort the arrays
    sort(eECal_i.begin(), eECal_i.end(), greater<double>());
    sort(eHCal_i.begin(), eHCal_i.end(), greater<double>());
    
    //Highest value in the array is the first element of the array
    fJetSpecific->m_maxEInEmTowers = eECal_i.front(); 
    fJetSpecific->m_maxEInHadTowers = eHCal_i.front();
    
    //n90 using the sorted list
    fJetSpecific->m_n90 = 0;
    double ediff = (eInHad + eInEm) * 0.9;
    for (unsigned i = 0; i < fTowerIds.size(); i++) {
      ediff = ediff - eECal_i[i] - eHCal_i[i];
      fJetSpecific->m_n90++;
      if (ediff <= 0) break; 
    }
    return true;
  }
}

CaloJet CaloJetMaker::makeCaloJet (const ProtoJet2& fProtojet) const {
  // construct towerIds
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

  CommonJetData common (fProtojet.px(), fProtojet.py(), fProtojet.pz(), 
			fProtojet.e(), fProtojet.p(), fProtojet.pt(), fProtojet.et(), fProtojet.m(), 
			fProtojet.phi(), fProtojet.eta(), fProtojet.y(), 
			fProtojet.numberOfConstituents());
  CaloJet::Specific specific;
  makeSpecific (*towerCollection, towerIds, &specific);

  return CaloJet (common, specific, towerIds);
}

