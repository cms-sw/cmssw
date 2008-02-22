/// Algorithm to convert transient protojets into persistent jets
/// Author: F.Ratnikov, UMd
/// Mar. 8, 2006
/// $Id: JetMaker.cc,v 1.27 2007/07/25 22:09:40 fedor Exp $

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "RecoJets/JetAlgorithms/interface/JetMaker.h"

using namespace std;
using namespace reco;

namespace {
  bool makeSpecific (const ProtoJet::Constituents& fTowers,
		     const CaloSubdetectorGeometry& fTowerGeometry,
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
    double jetArea = 0.;
    
    for (ProtoJet::Constituents::const_iterator towerCand = fTowers.begin(); towerCand != fTowers.end(); ++towerCand) {
      const Candidate* candidate = towerCand->get ();
      if (candidate) {
	const CaloTower* tower = CaloJet::caloTower (candidate).get ();
	if (tower) {
	  //Array of energy in EM Towers:
	  eECal_i.push_back(tower->emEnergy());
	  eInEm += tower->emEnergy();
	  //Array of energy in HCAL Towers:
	  eHCal_i.push_back(tower->hadEnergy()); 
	  eInHad += tower->hadEnergy();
	  
	  //  figure out contributions
	  switch (JetMaker::hcalSubdetector (tower->id().ieta())) {
	  case HcalBarrel:
	    eInHB += tower->hadEnergy(); 
	    eInHO += tower->outerEnergy();
	    eInEB += tower->emEnergy();
	    break;
	  case HcalEndcap:
	    eInHE += tower->hadEnergy();
	    eInEE += tower->emEnergy();
	    break;
	  case HcalForward:
	    eHadInHF += tower->hadEnergy();
	    eEmInHF += tower->emEnergy();
	    break;
	  default:
	    break;
	  }
	  // get area of the tower (++ minus --)
	  const CaloCellGeometry* geometry = fTowerGeometry.getGeometry(tower->id());
	  if (geometry) {
	    float dEta = fabs (geometry->getCorners() [0].eta() - geometry->getCorners() [2].eta());
	    float dPhi = fabs (geometry->getCorners() [0].phi() - geometry->getCorners() [2].phi());
	    jetArea += dEta * dPhi;
	  }
	  else {
	    std::cerr << "JetMaker::makeSpecific (CaloJet)-> Geometry for cell " << tower->id() << " can not be found. Ignoring cell" << std::endl;
	  }
	}
	else {
	  std::cerr << "JetMaker::makeSpecific (CaloJet)-> Referred CaloTower is not available in the event" << std::endl;
	}
      }
      else {
	std::cerr << "JetMaker::makeSpecific (CaloJet)-> Referred constituent is not available in the event" << std::endl;
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
    if (towerEnergy > 0) {
      fJetSpecific->mEnergyFractionHadronic = eInHad / towerEnergy;
      fJetSpecific->mEnergyFractionEm = eInEm / towerEnergy;
    }
    else { // HO only jet
      fJetSpecific->mEnergyFractionHadronic = 0; 
      fJetSpecific->mEnergyFractionEm = 0;
    }
    fJetSpecific->mTowersArea = jetArea;
    fJetSpecific->mMaxEInEmTowers = 0;
    fJetSpecific->mMaxEInHadTowers = 0;
    
    //Sort the arrays
    sort(eECal_i.begin(), eECal_i.end(), greater<double>());
    sort(eHCal_i.begin(), eHCal_i.end(), greater<double>());
    
    if (!fTowers.empty ()) {  
      //Highest value in the array is the first element of the array
      fJetSpecific->mMaxEInEmTowers = eECal_i.front(); 
      fJetSpecific->mMaxEInHadTowers = eHCal_i.front();
      
    }
    return true;
  }
  
  ///@@@ PFJET *************************
  bool makeSpecific (const ProtoJet::Constituents& fPFCandidates,		   
		     PFJet::Specific* fJetSpecific) {
    if (!fJetSpecific) return false;
    
    // 1.- Loop over PFCandidates, 
    // 2.- Get the corresponding PFCandidate
    // 3.- Calculate the different PFJet specific quantities

    float chargedHadronEnergy=0.;
    float neutralHadronEnergy=0.;
    float chargedEmEnergy=0.;
    float neutralEmEnergy=0.;
    float chargedMuEnergy=0.;
    int   chargedMultiplicity=0;
    int   neutralMultiplicity=0;
    int   muonMultiplicity=0;
    
    ProtoJet::Constituents::const_iterator constituent = fPFCandidates.begin();
    for (; constituent != fPFCandidates.end(); ++constituent) {
      const Candidate* candidate = constituent->get ();
      if (candidate) {
	const PFCandidate* pfCand = dynamic_cast<const PFCandidate*> (candidate);
	if (pfCand) {
	  switch ( PFCandidate::ParticleType (pfCand->particleId())) {
	  case PFCandidate::h:       // charged hadron
	    chargedHadronEnergy += pfCand->energy();
	    chargedMultiplicity++;
	    break;
	    
	  case PFCandidate::e:       // electron 
	    chargedEmEnergy += pfCand->energy(); 
	    chargedMultiplicity++;
	    break;
	    
	  case PFCandidate::mu:      // muon
	    chargedMuEnergy += pfCand->energy();
	    chargedMultiplicity++;
	    muonMultiplicity++;
	    break;
	    
	  case PFCandidate::gamma:   // photon
	    neutralEmEnergy += pfCand->energy();
	    neutralMultiplicity++;
	    break;
	    
	  case  PFCandidate::h0 :    // neutral hadron
	    neutralHadronEnergy += pfCand->energy();
	    neutralMultiplicity++;
	    break;
	  
	  default:
	    std::cerr << "JetMaker::makeSpecific (PFJetJet)-> Unknown PFCandidate::ParticleType: " << pfCand->particleId() << " is ignored" << std::endl;
	    break;
	  }
	}
	else {
	   std::cerr << "JetMaker::makeSpecific (PFJetJet)-> Referred constituent is not PFCandidate" << std::endl;
	}
      }
      else {
	std::cerr << "JetMaker::makeSpecific (PFJetJet)-> Referred constituent is not available in the event" << std::endl;
      }
    }
    fJetSpecific->mChargedHadronEnergy=chargedHadronEnergy;
    fJetSpecific->mNeutralHadronEnergy= neutralHadronEnergy;
    fJetSpecific->mChargedEmEnergy=chargedEmEnergy;
    fJetSpecific->mChargedMuEnergy=chargedMuEnergy;
    fJetSpecific->mNeutralEmEnergy=neutralEmEnergy;
    fJetSpecific->mChargedMultiplicity=chargedMultiplicity;
    fJetSpecific->mNeutralMultiplicity=neutralMultiplicity;
    fJetSpecific->mMuonMultiplicity=muonMultiplicity;
    return true;
  }
 
  bool makeSpecific (const ProtoJet::Constituents& fMcParticles, 
		     GenJet::Specific* fJetSpecific) {
    for (ProtoJet::Constituents::const_iterator genCand = fMcParticles.begin(); genCand != fMcParticles.end(); ++genCand) {
      CandidateBaseRef master = CandidateBaseRef (*genCand); // ref to this
      if ((*genCand)->hasMasterClone ()) master = (*genCand)->masterClone();
      const Candidate* candidate = master.get ();
      if (candidate) {
	const GenParticleCandidate* genParticle = GenJet::genParticle (candidate);
	if (genParticle) {
	  double e = genParticle->energy();
	  switch (abs (genParticle->pdgId ())) {
	  case 22: // photon
	  case 11: // e
	    fJetSpecific->m_EmEnergy += e;
	    break;
	  case 211: // pi
	  case 321: // K
	  case 130: // KL
	  case 2212: // p
	  case 2112: // n
	    fJetSpecific->m_HadEnergy += e;
	    break;
	  case 13: // muon
	  case 12: // nu_e
	  case 14: // nu_mu
	  case 16: // nu_tau
	    
	    fJetSpecific->m_InvisibleEnergy += e;
	    break;
	  default: 
	    fJetSpecific->m_AuxiliaryEnergy += e;
	  }
	}
	else {
	  std::cerr << "JetMaker::makeSpecific (GenJet)-> Referred  GenParticleCandidate is not available in the event" << std::endl;
	}
      }
      else {
	std::cerr << "JetMaker::makeSpecific (GenJet)-> Referred constituent is not available in the event" << std::endl;
      }
    }
    return true;
  }
} // unnamed namespace

BasicJet JetMaker::makeBasicJet (const ProtoJet& fProtojet) const {
  return  BasicJet (fProtojet.p4(), reco::Particle::Point (0, 0, 0), fProtojet.getTowerList());
}


CaloJet JetMaker::makeCaloJet (const ProtoJet& fProtojet, const CaloSubdetectorGeometry& fTowerGeometry) const {
  CaloJet::Specific specific;
  makeSpecific (fProtojet.getTowerList(), fTowerGeometry, &specific);
  return CaloJet (fProtojet.p4(), specific, fProtojet.getTowerList());
}

PFJet JetMaker::makePFJet (const ProtoJet& fProtojet) const {
  PFJet::Specific specific;
  makeSpecific (fProtojet.getTowerList(), &specific);
  return PFJet (fProtojet.p4(), specific, fProtojet.getTowerList());
}

GenJet JetMaker::makeGenJet (const ProtoJet& fProtojet) const {
  GenJet::Specific specific;
  makeSpecific (fProtojet.getTowerList(), &specific);
  return GenJet (fProtojet.p4(), specific, fProtojet.getTowerList());
}

GenericJet JetMaker::makeGenericJet (const ProtoJet& fProtojet) const {
  // count constituents
  std::vector<CandidateBaseRef>  constituents;
  // fill nothing into consituents for now
  return  GenericJet (fProtojet.p4(), reco::Particle::Point (0, 0, 0), constituents);
}

HcalSubdetector JetMaker::hcalSubdetector (int fEta) {
  static const HcalTopology topology;
  int eta = abs (fEta);
  if (eta <= topology.lastHBRing()) return HcalBarrel;
  else if (eta <= topology.lastHERing()) return HcalEndcap;
  else if (eta <= topology.lastHFRing()) return HcalForward;
  return HcalEmpty;
}
