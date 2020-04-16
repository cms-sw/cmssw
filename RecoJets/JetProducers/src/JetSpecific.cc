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

////////////////////////////////////////////////////////////////////////////////
// implementation of global functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
// Overloaded methods to write out specific types

// CaloJet
void reco::writeSpecific(reco::CaloJet& jet,
                         reco::Particle::LorentzVector const& p4,
                         reco::Particle::Point const& point,
                         std::vector<reco::CandidatePtr> const& constituents,
                         edm::EventSetup const& c) {
  // Get geometry
  edm::ESHandle<CaloGeometry> geometry;
  c.get<CaloGeometryRecord>().get(geometry);
  const CaloSubdetectorGeometry* towerGeometry =
      geometry->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);

  edm::ESHandle<HcalTopology> topology;
  c.get<HcalRecNumberingRecord>().get(topology);

  // Make the specific
  reco::CaloJet::Specific specific;
  makeSpecific(constituents, towerGeometry, &specific, *topology);
  // Set the calo jet
  jet = reco::CaloJet(p4, point, specific, constituents);
}

// BasicJet
void reco::writeSpecific(reco::BasicJet& jet,
                         reco::Particle::LorentzVector const& p4,
                         reco::Particle::Point const& point,
                         std::vector<reco::CandidatePtr> const& constituents,
                         edm::EventSetup const& c) {
  jet = reco::BasicJet(p4, point, constituents);
}

// GenJet
void reco::writeSpecific(reco::GenJet& jet,
                         reco::Particle::LorentzVector const& p4,
                         reco::Particle::Point const& point,
                         std::vector<reco::CandidatePtr> const& constituents,
                         edm::EventSetup const& c) {
  // Make the specific
  reco::GenJet::Specific specific;
  makeSpecific(constituents, &specific);
  // Set to the jet
  jet = reco::GenJet(p4, point, specific, constituents);
}

// PFJet
void reco::writeSpecific(reco::PFJet& jet,
                         reco::Particle::LorentzVector const& p4,
                         reco::Particle::Point const& point,
                         std::vector<reco::CandidatePtr> const& constituents,
                         edm::EventSetup const& c,
                         edm::ValueMap<float> const* weights) {
  // Make the specific
  reco::PFJet::Specific specific;
  makeSpecific(constituents, &specific, weights);
  // now make jet charge
  int charge = 0.;
  for (std::vector<reco::CandidatePtr>::const_iterator ic = constituents.begin(), icend = constituents.end();
       ic != icend;
       ++ic) {
    float weight = (weights != nullptr) ? (*weights)[*ic] : 1.0;
    charge += (*ic)->charge() * weight;
  }
  jet = reco::PFJet(p4, point, specific, constituents);
  jet.setCharge(charge);
}

// TrackJet
void reco::writeSpecific(reco::TrackJet& jet,
                         reco::Particle::LorentzVector const& p4,
                         reco::Particle::Point const& point,
                         std::vector<reco::CandidatePtr> const& constituents,
                         edm::EventSetup const& c) {
  jet = reco::TrackJet(p4, point, constituents);
}

// PFClusterJet
void reco::writeSpecific(reco::PFClusterJet& jet,
                         reco::Particle::LorentzVector const& p4,
                         reco::Particle::Point const& point,
                         std::vector<reco::CandidatePtr> const& constituents,
                         edm::EventSetup const& c) {
  jet = reco::PFClusterJet(p4, point, constituents);
}

//______________________________________________________________________________
bool reco::makeSpecific(std::vector<reco::CandidatePtr> const& towers,
                        const CaloSubdetectorGeometry* towerGeometry,
                        CaloJet::Specific* caloJetSpecific,
                        const HcalTopology& topology) {
  if (nullptr == caloJetSpecific)
    return false;

  // 1.- Loop over the tower Ids,
  // 2.- Get the corresponding CaloTower
  // 3.- Calculate the different CaloJet specific quantities
  std::vector<double> eECal_i;
  std::vector<double> eHCal_i;
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

  std::vector<reco::CandidatePtr>::const_iterator itTower;
  for (itTower = towers.begin(); itTower != towers.end(); ++itTower) {
    if (itTower->isNull() || !itTower->isAvailable()) {
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
      switch (reco::hcalSubdetector(tower->id().ieta(), topology)) {
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
      auto geometry = towerGeometry->getGeometry(tower->id());
      if (geometry) {
        jetArea += geometry->etaSpan() * geometry->phiSpan();
      } else {
        edm::LogWarning("DataNotFound") << "reco::makeCaloJetSpecific: Geometry for cell " << tower->id()
                                        << " can not be found. Ignoring cell\n";
      }
    } else {
      edm::LogWarning("DataNotFound") << "reco::makeCaloJetSpecific: Constituent is not of "
                                      << "CaloTower type\n";
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
  } else {  // HO only jet
    caloJetSpecific->mEnergyFractionHadronic = 1.;
    caloJetSpecific->mEnergyFractionEm = 0.;
  }
  caloJetSpecific->mTowersArea = jetArea;
  caloJetSpecific->mMaxEInEmTowers = 0;
  caloJetSpecific->mMaxEInHadTowers = 0;

  //Sort the arrays
  sort(eECal_i.begin(), eECal_i.end(), std::greater<double>());
  sort(eHCal_i.begin(), eHCal_i.end(), std::greater<double>());

  if (!towers.empty()) {
    //Highest value in the array is the first element of the array
    caloJetSpecific->mMaxEInEmTowers = eECal_i.front();
    caloJetSpecific->mMaxEInHadTowers = eHCal_i.front();
  }

  return true;
}

//______________________________________________________________________________
bool reco::makeSpecific(std::vector<reco::CandidatePtr> const& particles,
                        PFJet::Specific* pfJetSpecific,
                        edm::ValueMap<float> const* weights) {
  if (nullptr == pfJetSpecific)
    return false;

  // 1.- Loop over PFCandidates,
  // 2.- Get the corresponding PFCandidate
  // 3.- Calculate the different PFJet specific quantities

  float chargedHadronEnergy = 0.;
  float neutralHadronEnergy = 0.;
  float photonEnergy = 0.;
  float electronEnergy = 0.;
  float muonEnergy = 0.;
  float HFHadronEnergy = 0.;
  float HFEMEnergy = 0.;
  float chargedHadronMultiplicity = 0;
  float neutralHadronMultiplicity = 0;
  float photonMultiplicity = 0;
  float electronMultiplicity = 0;
  float muonMultiplicity = 0;
  float HFHadronMultiplicity = 0;
  float HFEMMultiplicity = 0;

  float chargedEmEnergy = 0.;
  float neutralEmEnergy = 0.;
  float chargedMuEnergy = 0.;
  float chargedMultiplicity = 0;
  float neutralMultiplicity = 0;

  float HOEnergy = 0.;

  std::vector<reco::CandidatePtr>::const_iterator itParticle;
  for (itParticle = particles.begin(); itParticle != particles.end(); ++itParticle) {
    if (itParticle->isNull() || !itParticle->isAvailable()) {
      edm::LogWarning("DataNotFound") << " JetSpecific: PF Particle is invalid\n";
      continue;
    }
    const Candidate* pfCand = itParticle->get();
    if (pfCand) {
      const PFCandidate* pfCandCast = dynamic_cast<const PFCandidate*>(pfCand);
      float weight = (weights != nullptr) ? (*weights)[*itParticle] : 1.0;
      if (pfCandCast)
        HOEnergy += pfCandCast->hoEnergy() * weight;

      switch (std::abs(pfCand->pdgId())) {
        case 211:  //PFCandidate::h:       // charged hadron
          chargedHadronEnergy += pfCand->energy() * weight;
          chargedHadronMultiplicity += weight;
          chargedMultiplicity += weight;
          break;

        case 130:  //PFCandidate::h0 :    // neutral hadron
          neutralHadronEnergy += pfCand->energy() * weight;
          neutralHadronMultiplicity += weight;
          neutralMultiplicity += weight;
          break;

        case 22:  //PFCandidate::gamma:   // photon
          photonEnergy += pfCand->energy() * weight;
          photonMultiplicity += weight;
          neutralEmEnergy += pfCand->energy() * weight;
          neutralMultiplicity += weight;
          break;

        case 11:  // PFCandidate::e:       // electron
          electronEnergy += pfCand->energy() * weight;
          electronMultiplicity += weight;
          chargedEmEnergy += pfCand->energy() * weight;
          chargedMultiplicity += weight;
          break;

        case 13:  //PFCandidate::mu:      // muon
          muonEnergy += pfCand->energy() * weight;
          muonMultiplicity += weight;
          chargedMuEnergy += pfCand->energy() * weight;
          chargedMultiplicity += weight;
          break;

        case 1:  // PFCandidate::h_HF :    // hadron in HF
          HFHadronEnergy += pfCand->energy() * weight;
          HFHadronMultiplicity += weight;
          neutralHadronEnergy += pfCand->energy() * weight;
          neutralMultiplicity += weight;
          break;

        case 2:  //PFCandidate::egamma_HF :    // electromagnetic in HF
          HFEMEnergy += pfCand->energy() * weight;
          HFEMMultiplicity += weight;
          neutralEmEnergy += pfCand->energy() * weight;
          neutralMultiplicity += weight;
          break;

        default:
          edm::LogWarning("DataNotFound")
              << "reco::makePFJetSpecific: Unknown PFCandidate::ParticleType: " << pfCand->pdgId() << " is ignored\n";
          break;
      }
    } else {
      edm::LogWarning("DataNotFound") << "reco::makePFJetSpecific: Referred constituent is not "
                                      << "a PFCandidate\n";
    }
  }

  pfJetSpecific->mChargedHadronEnergy = chargedHadronEnergy;
  pfJetSpecific->mNeutralHadronEnergy = neutralHadronEnergy;
  pfJetSpecific->mPhotonEnergy = photonEnergy;
  pfJetSpecific->mElectronEnergy = electronEnergy;
  pfJetSpecific->mMuonEnergy = muonEnergy;
  pfJetSpecific->mHFHadronEnergy = HFHadronEnergy;
  pfJetSpecific->mHFEMEnergy = HFEMEnergy;

  pfJetSpecific->mChargedHadronMultiplicity = std::round(chargedHadronMultiplicity);
  pfJetSpecific->mNeutralHadronMultiplicity = std::round(neutralHadronMultiplicity);
  pfJetSpecific->mPhotonMultiplicity = std::round(photonMultiplicity);
  pfJetSpecific->mElectronMultiplicity = std::round(electronMultiplicity);
  pfJetSpecific->mMuonMultiplicity = std::round(muonMultiplicity);
  pfJetSpecific->mHFHadronMultiplicity = std::round(HFHadronMultiplicity);
  pfJetSpecific->mHFEMMultiplicity = std::round(HFEMMultiplicity);

  pfJetSpecific->mChargedEmEnergy = chargedEmEnergy;
  pfJetSpecific->mChargedMuEnergy = chargedMuEnergy;
  pfJetSpecific->mNeutralEmEnergy = neutralEmEnergy;
  pfJetSpecific->mChargedMultiplicity = std::round(chargedMultiplicity);
  pfJetSpecific->mNeutralMultiplicity = std::round(neutralMultiplicity);

  pfJetSpecific->mHOEnergy = HOEnergy;

  return true;
}

//______________________________________________________________________________
bool reco::makeSpecific(std::vector<reco::CandidatePtr> const& mcparticles, GenJet::Specific* genJetSpecific) {
  if (nullptr == genJetSpecific)
    return false;

  std::vector<reco::CandidatePtr>::const_iterator itMcParticle = mcparticles.begin();
  for (; itMcParticle != mcparticles.end(); ++itMcParticle) {
    if (itMcParticle->isNull() || !itMcParticle->isAvailable()) {
      edm::LogWarning("DataNotFound") << " JetSpecific: MC Particle is invalid\n";
      continue;
    }

    const Candidate* candidate = itMcParticle->get();
    if (candidate->hasMasterClone())
      candidate = candidate->masterClone().get();
    //const GenParticle* genParticle = GenJet::genParticle(candidate);

    if (candidate) {
      double e = candidate->energy();

      // Legacy calo-like definitions
      switch (std::abs(candidate->pdgId())) {
        case 22:  // photon
        case 11:  // e
          genJetSpecific->m_EmEnergy += e;
          break;
        case 211:   // pi
        case 321:   // K
        case 130:   // KL
        case 2212:  // p
        case 2112:  // n
          genJetSpecific->m_HadEnergy += e;
          break;
        case 13:  // muon
        case 12:  // nu_e
        case 14:  // nu_mu
        case 16:  // nu_tau
          genJetSpecific->m_InvisibleEnergy += e;
          break;
        default:
          genJetSpecific->m_AuxiliaryEnergy += e;
      }

      // PF-like definitions
      switch (std::abs(candidate->pdgId())) {
        case 11:  //electron
          genJetSpecific->m_ChargedEmEnergy += e;
          ++(genJetSpecific->m_ChargedEmMultiplicity);
          break;
        case 13:  // muon
          genJetSpecific->m_MuonEnergy += e;
          ++(genJetSpecific->m_MuonMultiplicity);
          break;
        case 211:   //pi+-
        case 321:   //K
        case 2212:  //p
        case 3222:  //Sigma+
        case 3112:  //Sigma-
        case 3312:  //Xi-
        case 3334:  //Omega-
          genJetSpecific->m_ChargedHadronEnergy += e;
          ++(genJetSpecific->m_ChargedHadronMultiplicity);
          break;
        case 310:   //KS0
        case 130:   //KL0
        case 3122:  //Lambda0
        case 3212:  //Sigma0
        case 3322:  //Xi0
        case 2112:  //n0
          genJetSpecific->m_NeutralHadronEnergy += e;
          ++(genJetSpecific->m_NeutralHadronMultiplicity);
          break;
        case 22:  //photon
          genJetSpecific->m_NeutralEmEnergy += e;
          ++(genJetSpecific->m_NeutralEmMultiplicity);
          break;
      }
    }  // end if found a candidate
    else {
      edm::LogWarning("DataNotFound") << "reco::makeGenJetSpecific: Referred  GenParticleCandidate "
                                      << "is not available in the event\n";
    }
  }  // end for loop over MC particles

  return true;
}

//______________________________________________________________________________
HcalSubdetector reco::hcalSubdetector(int iEta, const HcalTopology& topology) {
  int eta = std::abs(iEta);
  if (eta <= topology.lastHBRing())
    return HcalBarrel;
  else if (eta <= topology.lastHERing())
    return HcalEndcap;
  else if (eta <= topology.lastHFRing())
    return HcalForward;
  return HcalEmpty;
}
