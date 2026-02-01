// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      PatFromScoutingJetProducer
//
/**\class PatFromScoutingJetProducer PatFromScoutingJetProducer.cc PhysicsTools/PatFromScouting/plugins/PatFromScoutingJetProducer.cc

 Description: Converts Run3ScoutingPFJet to pat::Jet

 Implementation:
     Creates pat::Jet with PFSpecific from scouting data, adds b-tagging discriminators
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Thu, 05 Dec 2024 15:27:09 GMT
//
//

#include <memory>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"

class PatFromScoutingJetProducer : public edm::stream::EDProducer<> {
public:
  explicit PatFromScoutingJetProducer(const edm::ParameterSet&);
  ~PatFromScoutingJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingPFJetCollection> jetToken_;
};

PatFromScoutingJetProducer::PatFromScoutingJetProducer(const edm::ParameterSet& iConfig)
    : jetToken_(consumes<Run3ScoutingPFJetCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<pat::JetCollection>();
}

void PatFromScoutingJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto patJets = std::make_unique<pat::JetCollection>();

  const auto& scoutingJets = iEvent.get(jetToken_);

  for (const auto& sJet : scoutingJets) {
    float px = sJet.pt() * std::cos(sJet.phi());
    float py = sJet.pt() * std::sin(sJet.phi());
    float pz = sJet.pt() * std::sinh(sJet.eta());
    float energy = std::sqrt(px * px + py * py + pz * pz + sJet.m() * sJet.m());

    reco::Particle::LorentzVector p4(px, py, pz, energy);

    reco::PFJet::Specific pfSpecific;
    pfSpecific.mChargedHadronEnergy = sJet.chargedHadronEnergy();
    pfSpecific.mNeutralHadronEnergy = sJet.neutralHadronEnergy();
    pfSpecific.mPhotonEnergy = sJet.photonEnergy();
    pfSpecific.mElectronEnergy = sJet.electronEnergy();
    pfSpecific.mMuonEnergy = sJet.muonEnergy();
    pfSpecific.mHFHadronEnergy = sJet.HFHadronEnergy();
    pfSpecific.mHFEMEnergy = sJet.HFEMEnergy();

    pfSpecific.mChargedHadronMultiplicity = sJet.chargedHadronMultiplicity();
    pfSpecific.mNeutralHadronMultiplicity = sJet.neutralHadronMultiplicity();
    pfSpecific.mPhotonMultiplicity = sJet.photonMultiplicity();
    pfSpecific.mElectronMultiplicity = sJet.electronMultiplicity();
    pfSpecific.mMuonMultiplicity = sJet.muonMultiplicity();
    pfSpecific.mHFHadronMultiplicity = sJet.HFHadronMultiplicity();
    pfSpecific.mHFEMMultiplicity = sJet.HFEMMultiplicity();

    pfSpecific.mHOEnergy = sJet.HOEnergy();

    pfSpecific.mChargedEmEnergy = sJet.electronEnergy();
    pfSpecific.mChargedMuEnergy = sJet.muonEnergy();
    pfSpecific.mNeutralEmEnergy = sJet.photonEnergy() + sJet.HFEMEnergy();

    int chargedMultiplicity =
        sJet.chargedHadronMultiplicity() + sJet.electronMultiplicity() + sJet.muonMultiplicity();
    int neutralMultiplicity =
        sJet.neutralHadronMultiplicity() + sJet.photonMultiplicity() + sJet.HFHadronMultiplicity() + sJet.HFEMMultiplicity();

    pfSpecific.mChargedMultiplicity = chargedMultiplicity;
    pfSpecific.mNeutralMultiplicity = neutralMultiplicity;

    reco::PFJet pfJet(p4, math::XYZPoint(0, 0, 0), pfSpecific);
    pfJet.setJetArea(sJet.jetArea());

    pat::Jet patJet(pfJet);

    patJet.addBDiscriminatorPair(std::make_pair("pfCombinedSecondaryVertexV2BJetTags", sJet.csv()));
    patJet.addBDiscriminatorPair(std::make_pair("pfDeepCSVJetTags:probb", sJet.mvaDiscriminator()));

    patJet.addUserFloat("csv", sJet.csv());
    patJet.addUserFloat("mvaDiscriminator", sJet.mvaDiscriminator());

    patJets->push_back(patJet);
  }

  iEvent.put(std::move(patJets));
}

void PatFromScoutingJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingPFPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PatFromScoutingJetProducer);
