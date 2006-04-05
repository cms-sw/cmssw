/// Algorithm to convert transient protojets into persistent jets
/// Author: F.Ratnikov, UMd
/// Mar. 8, 2006
/// $Id: JetMaker.h,v 1.2 2006/03/31 20:57:51 fedor Exp $

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "CLHEP/HepMC/GenEvent.h"

#include "RecoJets/JetAlgorithms/interface/JetMaker.h"

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
      eECal_i.push_back(aTower->emEnergy());
      eInEm += aTower->emEnergy();
      //Array of energy in HCAL Towers:
      eHCal_i.push_back(aTower->hadEnergy()); 
      eInHad += aTower->hadEnergy();
      
      eInHO += aTower->outerEnergy();
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
  
  bool makeSpecific (const std::vector<const HepMC::GenParticle*>& fMcParticles, 
		     GenJet::Specific* fJetSpecific) {
    std::vector<const HepMC::GenParticle*>::const_iterator it = fMcParticles.begin ();
    for (; it != fMcParticles.end (); it++) {
      const HepMC::GenParticle* genParticle = *it;
      switch (abs (genParticle->pdg_id ())) {
      case 22: // photon
      case 11: // e
	fJetSpecific->m_EmEnergy += genParticle->momentum().e ();
	break;
      case 211: // pi
      case 321: // K
      case 130: // KL
      case 2212: // p
      case 2112: // n
	  fJetSpecific->m_HadEnergy += genParticle->momentum().e ();
	break;
      case 13: // muon
      case 12: // nu_e
      case 14: // nu_mu
      case 16: // nu_tau

	fJetSpecific->m_InvisibleEnergy += genParticle->momentum().e ();
	break;
      default: 
	std::cerr << "makeSpecific-> Unknown stable particle " << genParticle->pdg_id () << std::endl;
      }
    }
    return true;
  }
}

bool JetMaker::convertableToCaloJet (const ProtoJet& fProtojet) const {
  const ProtoJet::Candidates* towers = &fProtojet.getTowerList();
  ProtoJet::Candidates::const_iterator tower = towers->begin ();
  for (; tower != towers->end (); tower++) {
    edm::Ref<CaloTowerCollection> towerRef = component<CaloTowerRef>::get (**tower);
    if (towerRef.isNull ()) return false; 
    break; // do not check all constituents
  }
  return true;
}

CaloJet JetMaker::makeCaloJet (const ProtoJet& fProtojet) const {
  // construct towerIds
  std::vector<CaloTowerDetId> towerIds;
  const CaloTowerCollection* towerCollection = 0;
  const ProtoJet::Candidates* towers = &fProtojet.getTowerList();
  towerIds.reserve (towers->size ());
  ProtoJet::Candidates::const_iterator tower = towers->begin ();
  for (; tower != towers->end (); tower++) {
    edm::Ref<CaloTowerCollection> towerRef = component<CaloTowerRef>::get (**tower);
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
      cerr << "CaloJetMaker::makeCaloJet-> Constituent candidate is not compatible with CaloTowerCandidate type" << std::endl;
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

bool JetMaker::convertableToGenJet (const ProtoJet& fProtojet) const {
  const ProtoJet::Candidates* towers = &fProtojet.getTowerList();
  ProtoJet::Candidates::const_iterator mcCandidate = towers->begin ();
  for (; mcCandidate != towers->end (); mcCandidate++) {
    const HepMC::GenParticle* genParticle = component<HepMCCandidate::GenParticleRef>::get (**mcCandidate);
    if (!genParticle) return false;
    break; // do not check all constituents
  }
  return true;
}


GenJet JetMaker::makeGenJet (const ProtoJet& fProtojet) const {
  const ProtoJet::Candidates* towers = &fProtojet.getTowerList();
  // construct MC barcodes
  std::vector<const HepMC::GenParticle*> mcParticles;
  mcParticles.reserve (towers->size ());
  std::vector<int> barcodes;
  barcodes.reserve (towers->size ());
  ProtoJet::Candidates::const_iterator mcCandidate = towers->begin ();
  for (; mcCandidate != towers->end (); mcCandidate++) {
    const HepMC::GenParticle* genParticle = component<HepMCCandidate::GenParticleRef>::get (**mcCandidate);
    if (genParticle) {
      mcParticles.push_back (genParticle);
      barcodes.push_back (genParticle->barcode ());
    }
    else {
      std::cerr << "JetMaker::makeGenJet-> Constituent candidate is not compatible with HepMCCandidate type" << std::endl;
    }
  }

  CommonJetData common (fProtojet.px(), fProtojet.py(), fProtojet.pz(), 
			fProtojet.e(), fProtojet.p(), fProtojet.pt(), fProtojet.et(), fProtojet.m(), 
			fProtojet.phi(), fProtojet.eta(), fProtojet.y(), 
			fProtojet.numberOfConstituents());
  GenJet::Specific specific;
  makeSpecific (mcParticles, &specific);

  return GenJet (common, specific, barcodes);
}

