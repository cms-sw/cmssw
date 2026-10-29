// -*- C++ -*-
//
// Package:    GeneratorInterface/GenFilters
// Class:      MCZllMesonPhotonFilter
//
// Selects Drell-Yan events that can fake a photon from a light meson decay.
// An event passes if all three conditions are met:
//   * a stable particle with |pdgId| == particleID and pT > minPt whose mother
//     has |pdgId| in motherIDs (by default a photon from a pi0 or eta decay);
//   * the cluster made of the highest-pT such particle (the seed) and every other
//     stable particle with |pdgId| == particleID within clusterDeltaR of the seed
//     has pT > minClusterPt. The axis is the seed direction, so the result does not
//     depend on the order of the particles in the HepMC record;
//   * the leading e- and e+, or the leading mu- and mu+, have an invariant mass
//     in (minDileptonMass, maxDileptonMass). Leptons of any status are used.
//

#include <cmath>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class MCZllMesonPhotonFilter : public edm::global::EDFilter<> {
public:
  explicit MCZllMesonPhotonFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const int particleID_;
  const std::vector<int> motherIDs_;
  const float minPt_;
  const double clusterDeltaR_;
  const double minClusterPt_;
  const double minDileptonMass_;
  const double maxDileptonMass_;
};

MCZllMesonPhotonFilter::MCZllMesonPhotonFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("src"))),
      particleID_(iConfig.getParameter<int>("particleID")),
      motherIDs_(iConfig.getParameter<std::vector<int>>("motherIDs")),
      minPt_(iConfig.getParameter<double>("minPt")),
      clusterDeltaR_(iConfig.getParameter<double>("clusterDeltaR")),
      minClusterPt_(iConfig.getParameter<double>("minClusterPt")),
      minDileptonMass_(iConfig.getParameter<double>("minDileptonMass")),
      maxDileptonMass_(iConfig.getParameter<double>("maxDileptonMass")) {}

bool MCZllMesonPhotonFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  const HepMC::GenEvent* genEvent = iEvent.get(token_).GetEvent();

  auto fromMeson = [this](int motherID) {
    for (int id : motherIDs_) {
      if (motherID == id)
        return true;
    }
    return false;
  };

  // highest-pT candidate of each lepton charge and flavor
  struct Leading {
    float pt = -1.f;
    math::XYZTLorentzVector p4;
    bool found = false;
  };
  Leading eMinus, ePlus, muMinus, muPlus;

  bool foundPhoton = false;
  std::vector<math::XYZTLorentzVector> photons;
  unsigned int seedIndex = 0;
  float seedPt = -1.f;

  for (auto p = genEvent->particles_begin(); p != genEvent->particles_end(); ++p) {
    const HepMC::GenVertex* prodVertex = (*p)->production_vertex();
    if (prodVertex == nullptr || prodVertex->particles_in_size() == 0)
      continue;
    const HepMC::GenParticle* mother = *(prodVertex->particles_in_const_begin());
    if (mother == nullptr)
      continue;

    const int pdgId = (*p)->pdg_id();
    const HepMC::FourVector& mom = (*p)->momentum();
    const math::XYZTLorentzVector p4(mom.px(), mom.py(), mom.pz(), mom.e());

    if (std::abs(pdgId) == particleID_ && (*p)->status() == 1) {
      const bool isFromMeson = fromMeson(std::abs(mother->pdg_id()));
      if (isFromMeson && mom.perp() > minPt_)
        foundPhoton = true;
      if (isFromMeson && mom.perp() > seedPt) {
        seedPt = mom.perp();
        seedIndex = photons.size();
      }
      photons.push_back(p4);
    } else {
      Leading* lepton = nullptr;
      if (pdgId == 11)
        lepton = &eMinus;
      else if (pdgId == -11)
        lepton = &ePlus;
      else if (pdgId == 13)
        lepton = &muMinus;
      else if (pdgId == -13)
        lepton = &muPlus;
      if (lepton != nullptr && mom.perp() > lepton->pt) {
        lepton->pt = mom.perp();
        lepton->p4 = p4;
        lepton->found = true;
      }
    }
  }

  if (!foundPhoton)
    return false;

  const math::XYZTLorentzVector& seed = photons[seedIndex];
  math::XYZTLorentzVector cluster = seed;
  for (unsigned int j = 0; j < photons.size(); ++j) {
    if (j != seedIndex && reco::deltaR(seed, photons[j]) < clusterDeltaR_)
      cluster += photons[j];
  }
  if (!(cluster.pt() > minClusterPt_))
    return false;

  auto inMassWindow = [this](const Leading& a, const Leading& b) {
    if (!a.found || !b.found)
      return false;
    const float mass = (a.p4 + b.p4).mass();
    return mass > minDileptonMass_ && mass < maxDileptonMass_;
  };
  return inMassWindow(eMinus, ePlus) || inMassWindow(muMinus, muPlus);
}

void MCZllMesonPhotonFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("generator", "unsmeared"));
  desc.add<int>("particleID", 22)->setComment("|pdgId| of the particle from the meson decay");
  desc.add<std::vector<int>>("motherIDs", {111, 221})->setComment("|pdgId| of the allowed mothers");
  desc.add<double>("minPt", 5.)->setComment("pT threshold of the particle from the meson decay");
  desc.add<double>("clusterDeltaR", 0.1);
  desc.add<double>("minClusterPt", 10.);
  desc.add<double>("minDileptonMass", 75.);
  desc.add<double>("maxDileptonMass", 105.);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MCZllMesonPhotonFilter);
