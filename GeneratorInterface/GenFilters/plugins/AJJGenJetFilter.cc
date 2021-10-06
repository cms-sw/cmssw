// -*- C++ -*-
//
// Package:    GeneratorInterface/GenFilters
// Class:      AJJGenJetFilter
//
/*

 Description: A filter to select events with one photon and 2 jets.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Hamed Bakhshian
//         Created:  Wed Oct 06 2021
//
//

// CMSSW include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <HepMC/GenVertex.h>

// C++ include files
#include <memory>
#include <map>
#include <vector>
#include <iostream>

using namespace edm;
using namespace std;
//
// class declaration
//

class AJJGenJetFilter : public edm::global::EDFilter<> {
public:
  explicit AJJGenJetFilter(const edm::ParameterSet& pset);
  ~AJJGenJetFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------memeber function----------------------
  const std::vector<const reco::GenJet*> filterGenJets(const vector<reco::GenJet>* jets) const;
  const std::vector<const reco::GenParticle*> filterGenLeptons(const std::vector<reco::GenParticle>* particles) const;
  const std::vector<const reco::GenParticle*> filterGenPhotons(const std::vector<reco::GenParticle>* particles) const;

  //**************************
  // Private Member data *****
private:
  // Dijet cut
  const double ptMin;
  const double etaMin;
  const double etaMax;
  const double minDeltaEta;
  const double maxDeltaEta;
  const double deltaRJetLep;
  const double maxPhotonEta;
  const double minPhotonPt;
  const double maxPhotonPt;
  const double mininvmass;

  // Input tags
  edm::EDGetTokenT<reco::GenJetCollection> m_GenJetCollection;
  edm::EDGetTokenT<reco::GenParticleCollection> m_GenParticleCollection;
};

AJJGenJetFilter::AJJGenJetFilter(const edm::ParameterSet& iConfig)
    : ptMin(iConfig.getUntrackedParameter<double>("minPt", 0)),
      etaMin(iConfig.getUntrackedParameter<double>("minEta", -4.5)),
      etaMax(iConfig.getUntrackedParameter<double>("maxEta", 4.5)),
      minDeltaEta(iConfig.getUntrackedParameter<double>("minDeltaEta", 0.0)),
      maxDeltaEta(iConfig.getUntrackedParameter<double>("maxDeltaEta", 99999.0)),
      deltaRJetLep(iConfig.getUntrackedParameter<double>("deltaRJetLep", 0.0)),
      maxPhotonEta(iConfig.getUntrackedParameter<double>("maxPhotonEta", 5)),
      minPhotonPt(iConfig.getUntrackedParameter<double>("minPhotonPt", 50)),
      maxPhotonPt(iConfig.getUntrackedParameter<double>("maxPhotonPt", 10000)),
      mininvmass(iConfig.getUntrackedParameter<double>("MinInvMass", 0.0)) {
  m_GenJetCollection = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("GenJetCollection"));
  m_GenParticleCollection = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"));

  edm::LogInfo("AJJGenJetFilter") << "Parameters:"
                                  << "jPtMin:" << ptMin << ",jEta:" << etaMin << "--" << etaMax << ",minDR(j,lep)"
                                  << deltaRJetLep << ",deltaEta(j1,j2)" << minDeltaEta << "--" << maxDeltaEta
                                  << "m(j1,j2) < " << mininvmass << "PhotonPt" << minPhotonPt << "--" << maxPhotonPt
                                  << "PhotonEta" << maxPhotonEta;
}

AJJGenJetFilter::~AJJGenJetFilter() {}

const vector<const reco::GenParticle*> AJJGenJetFilter::filterGenLeptons(
    const vector<reco::GenParticle>* particles) const {
  vector<const reco::GenParticle*> out;

  for (const auto& p : *particles) {
    int absPdgId = std::abs(p.pdgId());

    if (((absPdgId == 11) || (absPdgId == 13) || (absPdgId == 15)) && p.isHardProcess()) {
      out.push_back(&p);
    }
  }
  return out;
}

const vector<const reco::GenParticle*> AJJGenJetFilter::filterGenPhotons(
    const vector<reco::GenParticle>* particles) const {
  vector<const reco::GenParticle*> out;

  for (const auto& p : *particles) {
    int absPdgId = std::abs(p.pdgId());

    if ((absPdgId == 22) && p.isHardProcess()) {
      if (abs(p.eta()) < maxPhotonEta && p.pt() > minPhotonPt && p.pt() <= maxPhotonPt) {
        out.push_back(&p);
      } else {
        edm::LogInfo("AJJPhoton") << "photon rejected, pt:" << p.pt() << " , eta:" << p.eta();
      }
    }
  }

  return out;
}

const vector<const reco::GenJet*> AJJGenJetFilter::filterGenJets(const vector<reco::GenJet>* jets) const {
  vector<const reco::GenJet*> out;

  for (unsigned i = 0; i < jets->size(); i++) {
    const reco::GenJet* j = &((*jets)[i]);

    if (j->p4().pt() > ptMin && j->p4().eta() > etaMin && j->p4().eta() < etaMax) {
      out.push_back(j);
    } else {
      edm::LogInfo("AJJJets") << "Jet rejected, pt:" << j->p4().pt() << " eta:" << j->p4().eta();
    }
  }

  return out;
}

// ------------ method called to skim the data  ------------
bool AJJGenJetFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  using namespace edm;

  Handle<vector<reco::GenJet> > handleGenJets;
  iEvent.getByToken(m_GenJetCollection, handleGenJets);
  const vector<reco::GenJet>* genJets = handleGenJets.product();
  // Getting filtered generator jets
  vector<const reco::GenJet*> filGenJets = filterGenJets(genJets);

  Handle<reco::GenParticleCollection> genParticelesCollection;
  iEvent.getByToken(m_GenParticleCollection, genParticelesCollection);
  const vector<reco::GenParticle>* genParticles = genParticelesCollection.product();
  vector<const reco::GenParticle*> filGenLep = filterGenLeptons(genParticles);

  // Getting p4 of jet with no lepton
  vector<math::XYZTLorentzVector> genJetsWithoutLeptonsP4;
  unsigned int jetIdx = 0;
  unsigned int nGoodJets = 0;
  while (jetIdx < filGenJets.size()) {
    bool jetWhitoutLep = true;

    const math::XYZTLorentzVector& p4J = (filGenJets[jetIdx])->p4();
    for (unsigned int i = 0; i < filGenLep.size() && jetWhitoutLep; ++i) {
      if (reco::deltaR2((filGenLep[i])->p4(), p4J) < deltaRJetLep * deltaRJetLep)
        jetWhitoutLep = false;
    }
    if (jetWhitoutLep) {
      if (genJetsWithoutLeptonsP4.size() < 2) {
        genJetsWithoutLeptonsP4.push_back(p4J);
      }
      nGoodJets++;
    };
    ++jetIdx;
  }

  vector<const reco::GenParticle*> filGenPhotons = filterGenPhotons(genParticles);

  if (filGenPhotons.size() != 1) {
    edm::LogInfo("AJJPhoton") << "Events rejected, number of photons:" << filGenPhotons.size();
    return false;
  }

  if (ptMin < 0)
    return true;

  //If we do not find at least 2 jets veto the event
  if (nGoodJets < 2) {
    edm::LogInfo("AJJJets") << "Events rejected, number of jets:" << nGoodJets;
    return false;
  }

  double dEta = fabs(genJetsWithoutLeptonsP4[0].eta() - genJetsWithoutLeptonsP4[1].eta());
  float invMassLeadingJet = (genJetsWithoutLeptonsP4[0] + genJetsWithoutLeptonsP4[1]).M();

  if (dEta >= minDeltaEta && dEta <= maxDeltaEta && invMassLeadingJet > mininvmass) {
    return true;
  }

  edm::LogInfo("AJJJets") << "Events rejected, dEta:" << dEta << ", mjj:" << invMassLeadingJet;
  return false;
}

void AJJGenJetFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("GenJetCollection", edm::InputTag("ak4GenJetsNoNu"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.addUntracked<double>("minPt", 0)->setComment("If this is negative, no cut on jets is applied");
  desc.addOptionalUntracked<double>("minEta", -4.5);
  desc.addOptionalUntracked<double>("maxEta", 4.5);
  desc.addOptionalUntracked<double>("deltaRJetLep", 0.);
  desc.addOptionalUntracked<double>("minDeltaEta", 0.0);
  desc.addOptionalUntracked<double>("maxDeltaEta", 9999.0);
  desc.addOptionalUntracked<double>("MinInvMass", 0.0);
  desc.addUntracked<double>("maxPhotonEta", 5);
  desc.addUntracked<double>("minPhotonPt", 50);
  desc.addUntracked<double>("maxPhotonPt", 10000);
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(AJJGenJetFilter);
