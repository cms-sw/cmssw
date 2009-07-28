////////////////////////////////////////////////////////////////////////////////
//
// JetSpecific
// -----------
//
////////////////////////////////////////////////////////////////////////////////


#include "RecoJets/JetProducers/interface/JetSpecific.h"


#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <cmath>


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// implementation of global functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________    
// Overloaded methods to write out specific types


// CaloJet
void reco::writeSpecific(reco::CaloJet & jet,
			 reco::Particle::LorentzVector const & p4,
			 reco::Particle::Point const & point, 
			 std::vector<reco::CandidatePtr> const & constituents,
			 edm::EventSetup const & c  )
{
  // Get geometry
  edm::ESHandle<CaloGeometry> geometry;
  c.get<CaloGeometryRecord>().get(geometry);
  const CaloSubdetectorGeometry* towerGeometry = 
    geometry->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);

  // Make the specific
  reco::CaloJet::Specific specific;
  makeSpecific (constituents, *towerGeometry, &specific);
  // Set the calo jet
  jet = reco::CaloJet( p4, point, specific, constituents);  
}
  

    
// BasicJet
void reco::writeSpecific(reco::BasicJet  & jet,
			 reco::Particle::LorentzVector const & p4,
			 reco::Particle::Point const & point, 
			 std::vector<reco::CandidatePtr> const & constituents,
			 edm::EventSetup const & c  )
{
  jet = reco::BasicJet( p4, point, constituents);  
}
    
// GenJet
void reco::writeSpecific(reco::GenJet  & jet,
			 reco::Particle::LorentzVector const & p4,
			 reco::Particle::Point const & point, 
			 std::vector<reco::CandidatePtr> const & constituents,
			 edm::EventSetup const & c  )
{

  // Make the specific
  reco::GenJet::Specific specific;
  makeSpecific (constituents, &specific);
  // Set to the jet
  jet = reco::GenJet( p4, point, specific, constituents);  
}
    
// PFJet
void reco::writeSpecific(reco::PFJet  & jet,
			 reco::Particle::LorentzVector const & p4,
			 reco::Particle::Point const & point, 
			 std::vector<reco::CandidatePtr> const & constituents,
			 edm::EventSetup const & c  )
{
  // Make the specific
  reco::PFJet::Specific specific;
  makeSpecific (constituents, &specific);
  jet = reco::PFJet( p4, point, specific, constituents);  
}
    




//______________________________________________________________________________
bool reco::makeSpecific(vector<reco::CandidatePtr> const & towers,
			const CaloSubdetectorGeometry& towerGeometry,
			CaloJet::Specific* caloJetSpecific)
{
  if (0==caloJetSpecific) return false;

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
  
  vector<reco::CandidatePtr>::const_iterator itTower;
  for (itTower=towers.begin();itTower!=towers.end();++itTower) {
    if ( itTower->isNull() || !itTower->isAvailable() ) { 
      edm::LogWarning("DataNotFound") << " JetSpecific: Tower is invalid\n";
      continue;
    }
    const CaloTower* tower = dynamic_cast<const CaloTower*>(itTower->get());
    if (tower) {
      //Array of energy in EM Towers:
      eECal_i.push_back(tower->emEnergy());
      eInEm += tower->emEnergy();
      //Array of energy in HCAL Towers:
      eHCal_i.push_back(tower->hadEnergy()); 
      eInHad += tower->hadEnergy();
      
      //  figure out contributions
      switch (reco::hcalSubdetector(tower->id().ieta())) {
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
      const CaloCellGeometry* geometry = towerGeometry.getGeometry(tower->id());
      if (geometry) {
	float dEta = fabs(geometry->getCorners()[0].eta()-
			  geometry->getCorners()[2].eta());
	float dPhi = fabs(geometry->getCorners()[0].phi() -
			  geometry->getCorners()[2].phi());
	jetArea += dEta * dPhi;
      }
      else {
	edm::LogWarning("DataNotFound") <<"reco::makeCaloJetSpecific: Geometry for cell "
					<<tower->id()<<" can not be found. Ignoring cell\n";
      }
    }
    else {
      edm::LogWarning("DataNotFound")<<"reco::makeCaloJetSpecific: Constituent is not of "
				     <<"CaloTower type\n";
    }
  }

  double towerEnergy = eInHad + eInEm;
  caloJetSpecific->mHadEnergyInHO = eInHO;
  caloJetSpecific->mHadEnergyInHB = eInHB;
  caloJetSpecific->mHadEnergyInHE = eInHE;
  caloJetSpecific->mHadEnergyInHF = eHadInHF;
  caloJetSpecific->mEmEnergyInHF = eEmInHF;
  caloJetSpecific->mEmEnergyInEB = eInEB;
  caloJetSpecific->mEmEnergyInEE = eInEE;
  if (towerEnergy > 0) {
    caloJetSpecific->mEnergyFractionHadronic = eInHad / towerEnergy;
    caloJetSpecific->mEnergyFractionEm = eInEm / towerEnergy;
  }
  else { // HO only jet
    caloJetSpecific->mEnergyFractionHadronic = 1.;
    caloJetSpecific->mEnergyFractionEm = 0.;
  }
  caloJetSpecific->mTowersArea = jetArea;
  caloJetSpecific->mMaxEInEmTowers = 0;
  caloJetSpecific->mMaxEInHadTowers = 0;
  
  //Sort the arrays
  sort(eECal_i.begin(), eECal_i.end(), greater<double>());
  sort(eHCal_i.begin(), eHCal_i.end(), greater<double>());
  
  if (!towers.empty()) {
    //Highest value in the array is the first element of the array
    caloJetSpecific->mMaxEInEmTowers  = eECal_i.front(); 
    caloJetSpecific->mMaxEInHadTowers = eHCal_i.front();
  }
  
  return true;
}


//______________________________________________________________________________
bool reco::makeSpecific(vector<reco::CandidatePtr> const & particles,	   
			PFJet::Specific* pfJetSpecific)
{
  if (0==pfJetSpecific) return false;
  
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
  
  vector<reco::CandidatePtr>::const_iterator itParticle;
  for (itParticle=particles.begin();itParticle!=particles.end();++itParticle){
    if ( itParticle->isNull() || !itParticle->isAvailable() ) { 
      edm::LogWarning("DataNotFound") << " JetSpecific: PF Particle is invalid\n";
      continue;
    }    
    const PFCandidate* pfCand = dynamic_cast<const PFCandidate*> (itParticle->get());
    if (pfCand) {
      switch (PFCandidate::ParticleType(pfCand->particleId())) {
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
      case PFCandidate::egamma_HF :    // electromagnetic in HF
	neutralEmEnergy += pfCand->energy();
	neutralMultiplicity++;
	break;
	
      case PFCandidate::h0 :    // neutral hadron
      case PFCandidate::h_HF :    // hadron in HF
	neutralHadronEnergy += pfCand->energy();
	neutralMultiplicity++;
	break;
	
      default:
	edm::LogWarning("DataNotFound") <<"reco::makePFJetSpecific: Unknown PFCandidate::ParticleType: "
					<<pfCand->particleId()<<" is ignored\n";
	break;
      }
    }
    else {
      edm::LogWarning("DataNotFound") <<"reco::makePFJetSpecific: Referred constituent is not "
				      <<"a PFCandidate\n";
    }
  }
  
  pfJetSpecific->mChargedHadronEnergy=chargedHadronEnergy;
  pfJetSpecific->mNeutralHadronEnergy= neutralHadronEnergy;
  pfJetSpecific->mChargedEmEnergy=chargedEmEnergy;
  pfJetSpecific->mChargedMuEnergy=chargedMuEnergy;
  pfJetSpecific->mNeutralEmEnergy=neutralEmEnergy;
  pfJetSpecific->mChargedMultiplicity=chargedMultiplicity;
  pfJetSpecific->mNeutralMultiplicity=neutralMultiplicity;
  pfJetSpecific->mMuonMultiplicity=muonMultiplicity;

  return true;
}


//______________________________________________________________________________
bool reco::makeSpecific(vector<reco::CandidatePtr> const & mcparticles, 
			GenJet::Specific* genJetSpecific)
{
  if (0==genJetSpecific) return false;

  vector<reco::CandidatePtr>::const_iterator itMcParticle=mcparticles.begin();
  for (;itMcParticle!=mcparticles.end();++itMcParticle) {
    if ( itMcParticle->isNull() || !itMcParticle->isAvailable() ) { 
      edm::LogWarning("DataNotFound") << " JetSpecific: MC Particle is invalid\n";
      continue;
    }
    const Candidate* candidate = itMcParticle->get();
    if (candidate->hasMasterClone()) candidate = candidate->masterClone().get();
    const GenParticle* genParticle = GenJet::genParticle(candidate);
    if (genParticle) {
      double e = genParticle->energy();
      switch (abs (genParticle->pdgId ())) {
      case 22: // photon
      case 11: // e
	genJetSpecific->m_EmEnergy += e;
	break;
      case 211: // pi
      case 321: // K
      case 130: // KL
      case 2212: // p
      case 2112: // n
	genJetSpecific->m_HadEnergy += e;
	break;
      case 13: // muon
      case 12: // nu_e
      case 14: // nu_mu
      case 16: // nu_tau
	
	genJetSpecific->m_InvisibleEnergy += e;
	break;
      default: 
	genJetSpecific->m_AuxiliaryEnergy += e;
      }
    }
    else {
      edm::LogWarning("DataNotFound") <<"reco::makeGenJetSpecific: Referred  GenParticleCandidate "
				      <<"is not available in the event\n";
    }
  }
  
  return true;
}


//______________________________________________________________________________
HcalSubdetector reco::hcalSubdetector(int iEta)
{
  static const HcalTopology topology;
  int eta = std::abs(iEta);
  if      (eta <= topology.lastHBRing()) return HcalBarrel;
  else if (eta <= topology.lastHERing()) return HcalEndcap;
  else if (eta <= topology.lastHFRing()) return HcalForward;
  return HcalEmpty;
}



