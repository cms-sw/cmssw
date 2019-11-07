#include "GeneratorInterface/GenFilters/plugins/VBFGenJetFilter.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <HepMC/GenVertex.h>

// ROOT includes
#include "TMath.h"

// C++ includes
#include <iostream>

using namespace edm;
using namespace std;

VBFGenJetFilter::VBFGenJetFilter(const edm::ParameterSet& iConfig)
    : oppositeHemisphere(iConfig.getUntrackedParameter<bool>("oppositeHemisphere", false)),
      leadJetsNoLepMass(iConfig.getUntrackedParameter<bool>("leadJetsNoLepMass", false)),
      ptMin(iConfig.getUntrackedParameter<double>("minPt", 20)),
      etaMin(iConfig.getUntrackedParameter<double>("minEta", -5.0)),
      etaMax(iConfig.getUntrackedParameter<double>("maxEta", 5.0)),
      minInvMass(iConfig.getUntrackedParameter<double>("minInvMass", 0.0)),
      maxInvMass(iConfig.getUntrackedParameter<double>("maxInvMass", 99999.0)),
      minDeltaPhi(iConfig.getUntrackedParameter<double>("minDeltaPhi", -1.0)),
      maxDeltaPhi(iConfig.getUntrackedParameter<double>("maxDeltaPhi", 99999.0)),
      minDeltaEta(iConfig.getUntrackedParameter<double>("minDeltaEta", -1.0)),
      maxDeltaEta(iConfig.getUntrackedParameter<double>("maxDeltaEta", 99999.0)),
      minLeadingJetsInvMass(iConfig.getUntrackedParameter<double>("minLeadingJetsInvMass", 0.0)),
      maxLeadingJetsInvMass(iConfig.getUntrackedParameter<double>("maxLeadingJetsInvMass", 99999.0)),
      deltaRJetLep(iConfig.getUntrackedParameter<double>("deltaRJetLep", 0.3)) {
  m_inputTag_GenJetCollection = consumes<reco::GenJetCollection>(
      iConfig.getUntrackedParameter<edm::InputTag>("inputTag_GenJetCollection", edm::InputTag("ak5GenJetsNoNu")));
  if (leadJetsNoLepMass)
    m_inputTag_GenParticleCollection = consumes<reco::GenParticleCollection>(
        iConfig.getUntrackedParameter<edm::InputTag>("genParticles", edm::InputTag("genParticles")));
}

VBFGenJetFilter::~VBFGenJetFilter() {}

vector<const reco::GenParticle*> VBFGenJetFilter::filterGenLeptons(const vector<reco::GenParticle>* particles) {
  vector<const reco::GenParticle*> out;

  for (const auto& p : *particles) {
    int absPdgId = std::abs(p.pdgId());

    if (((absPdgId == 11) || (absPdgId == 13) || (absPdgId == 15)) && p.isHardProcess()) {
      out.push_back(&p);
    }
  }
  return out;
}

vector<const reco::GenJet*> VBFGenJetFilter::filterGenJets(const vector<reco::GenJet>* jets) {
  vector<const reco::GenJet*> out;

  for (unsigned i = 0; i < jets->size(); i++) {
    const reco::GenJet* j = &((*jets)[i]);

    if (j->p4().pt() > ptMin && j->p4().eta() > etaMin && j->p4().eta() < etaMax) {
      out.push_back(j);
    }
  }

  return out;
}

// ------------ method called to skim the data  ------------
bool VBFGenJetFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<vector<reco::GenJet> > handleGenJets;
  iEvent.getByToken(m_inputTag_GenJetCollection, handleGenJets);
  const vector<reco::GenJet>* genJets = handleGenJets.product();

  // Getting filtered generator jets
  vector<const reco::GenJet*> filGenJets = filterGenJets(genJets);

  // If we do not find at least 2 jets veto the event
  if (filGenJets.size() < 2) {
    return false;
  }

  // Testing dijet mass
  if (leadJetsNoLepMass) {
    Handle<reco::GenParticleCollection> genParticelesCollection;
    iEvent.getByToken(m_inputTag_GenParticleCollection, genParticelesCollection);
    const vector<reco::GenParticle>* genParticles = genParticelesCollection.product();

    // Getting filtered generator muons
    vector<const reco::GenParticle*> filGenLep = filterGenLeptons(genParticles);

    // Getting p4 of jet with no lepton
    vector<math::XYZTLorentzVector> genJetsWithoutLeptonsP4;
    unsigned int jetIdx = 0;

    while (genJetsWithoutLeptonsP4.size() < 2 && jetIdx < filGenJets.size()) {
      bool jetWhitoutLep = true;

      const math::XYZTLorentzVector& p4J = (filGenJets[jetIdx])->p4();
      for (unsigned int i = 0; i < filGenLep.size() && jetWhitoutLep; ++i) {
        if (reco::deltaR2((filGenLep[i])->p4(), p4J) < deltaRJetLep * deltaRJetLep)
          jetWhitoutLep = false;
      }
      if (jetWhitoutLep)
        genJetsWithoutLeptonsP4.push_back(p4J);
      ++jetIdx;
    }

    // Checking the invariant mass of the leading jets
    if (genJetsWithoutLeptonsP4.size() < 2)
      return false;
    float invMassLeadingJet = (genJetsWithoutLeptonsP4[0] + genJetsWithoutLeptonsP4[1]).M();
    if (invMassLeadingJet > minLeadingJetsInvMass && invMassLeadingJet < maxLeadingJetsInvMass)
      return true;
    else
      return false;
  }

  for (unsigned a = 0; a < filGenJets.size(); a++) {
    for (unsigned b = a + 1; b < filGenJets.size(); b++) {
      const reco::GenJet* pA = filGenJets[a];
      const reco::GenJet* pB = filGenJets[b];

      // Getting the dijet vector
      math::XYZTLorentzVector diJet = pA->p4() + pB->p4();

      // Testing opposite hemispheres
      double dijetProd = pA->p4().eta() * pB->p4().eta();
      if (oppositeHemisphere && dijetProd >= 0) {
        continue;
      }

      // Testing dijet mass
      double invMass = diJet.mass();
      if (invMass <= minInvMass || invMass > maxInvMass) {
        continue;
      }

      // Testing dijet delta eta
      double dEta = fabs(pA->p4().eta() - pB->p4().eta());
      if (dEta <= minDeltaEta || dEta > maxDeltaEta) {
        continue;
      }

      // Testing dijet delta phi
      double dPhi = fabs(reco::deltaPhi(pA->p4().phi(), pB->p4().phi()));
      if (dPhi <= minDeltaPhi || dPhi > maxDeltaPhi) {
        continue;
      }

      return true;
    }
  }

  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(VBFGenJetFilter);
