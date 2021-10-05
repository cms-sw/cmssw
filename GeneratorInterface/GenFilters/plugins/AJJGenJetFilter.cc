#include "GeneratorInterface/GenFilters/plugins/AJJGenJetFilter.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <HepMC/GenVertex.h>

// ROOT includes
#include "TMath.h"
#include "TH1.h"
#include "TH2.h"

// C++ includes
#include <iostream>
#include <vector>

using namespace edm;
using namespace std;

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

vector<const reco::GenParticle*> AJJGenJetFilter::filterGenLeptons(const vector<reco::GenParticle>* particles) {
  vector<const reco::GenParticle*> out;

  for (const auto& p : *particles) {
    int absPdgId = std::abs(p.pdgId());

    if (((absPdgId == 11) || (absPdgId == 13) || (absPdgId == 15)) && p.isHardProcess()) {
      out.push_back(&p);
    }
  }
  return out;
}

vector<const reco::GenParticle*> AJJGenJetFilter::filterGenPhotons(const vector<reco::GenParticle>* particles) {
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

vector<const reco::GenJet*> AJJGenJetFilter::filterGenJets(const vector<reco::GenJet>* jets) {
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
bool AJJGenJetFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
//define this as a plug-in
DEFINE_FWK_MODULE(AJJGenJetFilter);
