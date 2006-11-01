/// Algorithm to convert transient protojets into persistent jets
/// Author: F.Ratnikov, UMd
/// Mar. 8, 2006
/// $Id: JetMaker.cc,v 1.12 2006/07/21 19:26:03 fedor Exp $

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
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
    double eInHE = 0.;
    double eHadInHF = 0.;
    double eEmInHF = 0.;
    double eInEB = 0.;
    double eInEE = 0.;
    
    for(vector<CaloTowerDetId>::const_iterator i = fTowerIds.begin(); i != fTowerIds.end(); ++i) {
      const CaloTower* aTower =  &*fTowers.find(*i);
      //Array of energy in EM Towers:
      eECal_i.push_back(aTower->emEnergy());
      eInEm += aTower->emEnergy();
      //Array of energy in HCAL Towers:
      eHCal_i.push_back(aTower->hadEnergy()); 
      eInHad += aTower->hadEnergy();
      
      eInHO += aTower->outerEnergy();

      //  figure out contributions
      bool hadIsDone = false;
      bool emIsDone = false;
      int icell = aTower->constituentsSize();
      while (--icell >= 0 && (!hadIsDone || !emIsDone)) {
	DetId id = aTower->constituent (icell);
	if (!hadIsDone && id.det () == DetId::Hcal) { // hcal cell
	  HcalSubdetector subdet = HcalDetId (id).subdet ();
	  if (subdet == HcalBarrel || subdet == HcalOuter) {
	    eInHB += aTower->hadEnergy(); 
	    eInHO += aTower->outerEnergy();
	  }
	  else if (subdet == HcalEndcap) {
	    eInHE += aTower->hadEnergy();
	  }
	  else if (subdet == HcalForward) {
	    eHadInHF += aTower->hadEnergy();
	    eEmInHF += aTower->emEnergy();
	    emIsDone = true;
	  }
	  hadIsDone = true;
	}
	else if (!emIsDone && id.det () == DetId::Ecal) { // ecal cell
	  EcalSubdetector subdet = EcalSubdetector (id.subdetId ());
	  if (subdet == EcalBarrel) {
	    eInEB += aTower->emEnergy();
	  }
	  else if (subdet == EcalEndcap) {
	    eInEE += aTower->emEnergy();
	  }
	  emIsDone = true;
	}
      }
    }
    double towerEnergy = eInHad + eInEm;
    fJetSpecific->mHadEnergyInHO = eInHO;
    fJetSpecific->mHadEnergyInHB = eInHB;
    fJetSpecific->mHadEnergyInHE = eInHE;
    fJetSpecific->mHadEnergyInHF = eHadInHF;
    fJetSpecific->mEmEnergyInHF = eEmInHF;
    fJetSpecific->mEmEnergyInEB = eInEB;
    fJetSpecific->mEmEnergyInEE = eInEE;
    fJetSpecific->mEnergyFractionHadronic = eInHad / towerEnergy;
    fJetSpecific->mEnergyFractionEm = eInEm / towerEnergy;
    fJetSpecific->mMaxEInEmTowers = 0;
    fJetSpecific->mMaxEInHadTowers = 0;
    fJetSpecific->mN90 = 0;
    
    //Sort the arrays
    sort(eECal_i.begin(), eECal_i.end(), greater<double>());
    sort(eHCal_i.begin(), eHCal_i.end(), greater<double>());
    
    if (!fTowerIds.empty ()) {  
      //Highest value in the array is the first element of the array
      fJetSpecific->mMaxEInEmTowers = eECal_i.front(); 
      fJetSpecific->mMaxEInHadTowers = eHCal_i.front();
      
      //n90 using the sorted list
      double ediff = (eInHad + eInEm) * 0.9;
      for (unsigned i = 0; i < fTowerIds.size(); i++) {
	ediff = ediff - eECal_i[i] - eHCal_i[i];
	fJetSpecific->mN90++;
	if (ediff <= 0) break; 
      }
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
        fJetSpecific->m_AuxiliaryEnergy += genParticle->momentum().e ();
      }
    }
    return true;
  }
}

BasicJet JetMaker::makeBasicJet (const ProtoJet& fProtojet) const {
  return BasicJet (fProtojet.p4(), reco::Particle::Point (0, 0, 0));
}


CaloJet JetMaker::makeCaloJet (const ProtoJet& fProtojet) const {
  // construct towerIds
  const ProtoJet::Candidates* towers = &fProtojet.getTowerList();
  std::vector<CaloTowerDetId> towerIds;
  towerIds.reserve (towers->size ());
  const CaloTowerCollection* towerCollection = 0;
  ProtoJet::Candidates::const_iterator tower = towers->begin ();
  for (; tower != towers->end (); tower++) {
    edm::Ref<CaloTowerCollection> towerRef = (*tower)->get<CaloTowerRef>();
    if (towerRef.isNonnull ()) { // valid
      const CaloTowerCollection* newproduct = towerRef.product ();
      if (!newproduct) {
	cerr << "CaloJetMaker::makeCaloJet (const ProtoJet& fProtojet) ERROR-> "
	     << "Can not find CaloTowerCollection for contributing CalTower: " <<  newproduct << endl;
      }
      if (!towerCollection) towerCollection  = newproduct;
      else if (towerCollection != newproduct) {
	cerr << "CaloJetMaker::makeCaloJet (const ProtoJet& fProtojet) ERROR-> "
	     << "CaloTower collection for tower is not the same. Previous: " <<  towerCollection 
	     << ", new: " << newproduct << endl;
      }
      towerIds.push_back (towerRef->id ());
    }
    else {
      cerr << "CaloJetMaker::makeCaloJet-> Constituent candidate is not compatible with CaloTowerCandidate type" << std::endl;
    }
  }
  CaloJet::Specific specific;
  if (towerCollection) makeSpecific (*towerCollection, towerIds, &specific);

  return CaloJet (fProtojet.p4(), specific, towerIds);
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
   HepMCCandidate::GenParticleRef genParticle = 
    (*mcCandidate)->get<HepMCCandidate::GenParticleRef>();
    if (genParticle.isNonnull()) {
      mcParticles.push_back (& * genParticle);
      barcodes.push_back (genParticle->barcode ());
    }
    else {
      std::cerr << "JetMaker::makeGenJet-> Constituent candidate is not compatible with HepMCCandidate type" << std::endl;
    }
  }
  GenJet::Specific specific;
  makeSpecific (mcParticles, &specific);

  return GenJet (fProtojet.p4(), specific, barcodes);
}

