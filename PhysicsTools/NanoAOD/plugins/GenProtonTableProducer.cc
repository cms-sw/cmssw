/****************************************************************************
 *
 * This is a part of PPS offline software.
 * Authors:
 *   Laurent Forthomme
 *   Michael Pitt
 *
 ****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

class GenProtonTableProducer : public edm::stream::EDProducer<> {
public:
  explicit GenProtonTableProducer(const edm::ParameterSet&);
  ~GenProtonTableProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<reco::GenParticleCollection> prunedCandsToken_;
  const edm::EDGetTokenT<reco::GenParticleCollection> puCandsToken_;
  const StringCutObjectSelector<reco::Candidate> protonsCut_;
  const std::string table_name_;
  const double tolerance_;
};

GenProtonTableProducer::GenProtonTableProducer(const edm::ParameterSet& iConfig)
    : prunedCandsToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("srcPruned"))),
      puCandsToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("srcPUProtons"))),
      protonsCut_(iConfig.getParameter<std::string>("cut")),
      table_name_(iConfig.getParameter<std::string>("name")),
      tolerance_(iConfig.getParameter<double>("tolerance")) {}

void GenProtonTableProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  // define the variables
  std::vector<float> pts, pzs, vzs;
  std::vector<int> isPUs;
  // first loop over signal protons
  for (const auto& pruned_cand : iEvent.get(prunedCandsToken_)) {
    if (!protonsCut_(pruned_cand))
      continue;
    pts.emplace_back(pruned_cand.pt());
    pzs.emplace_back(pruned_cand.pz());
    vzs.emplace_back(pruned_cand.vz());
    isPUs.emplace_back(false);
  }
  // then loop over pruned candidates ; if already in signal protons, discard
  for (const auto& pu_cand : iEvent.get(puCandsToken_)) {
    if (!protonsCut_(pu_cand))
      continue;
    bool associated{false};
    for (size_t i = 0; i < pts.size(); ++i) {
      if (fabs(1. - pts.at(i) / pu_cand.pt()) < tolerance_ && fabs(1. - pzs.at(i) / pu_cand.pz()) < tolerance_) {
        associated = true;
        break;
      }
    }
    if (associated)
      continue;
    pts.emplace_back(pu_cand.pt());
    pzs.emplace_back(pu_cand.pz());
    vzs.emplace_back(pu_cand.vz());
    isPUs.emplace_back(true);
  }

  auto protons_table = std::make_unique<nanoaod::FlatTable>(isPUs.size(), table_name_, false);
  protons_table->addColumn<float>("pt", pts, "proton transverse momentum", nanoaod::FlatTable::FloatColumn, 10);
  protons_table->addColumn<float>("pz", pzs, "proton longitudinal momentum", nanoaod::FlatTable::FloatColumn, 10);
  protons_table->addColumn<float>(
      "vz", vzs, "proton vertex longitudinal coordinate", nanoaod::FlatTable::FloatColumn, 10);
  protons_table->addColumn<int>("isPU", isPUs, "pileup proton?", nanoaod::FlatTable::IntColumn, 10);
  iEvent.put(std::move(protons_table));
}

void GenProtonTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcPruned", edm::InputTag())
      ->setComment("input source for pruned gen-level particle candidates");
  desc.add<edm::InputTag>("srcPUProtons", edm::InputTag())->setComment("input source for pileup protons collection");
  desc.add<std::string>("cut", "")->setComment("proton kinematic selection");
  desc.add<std::string>("name", "GenProtons")->setComment("flat table name");
  desc.add<double>("tolerance", 1.e-3)
      ->setComment("relative difference between the signal and pileup protons pt and pz");
  descriptions.add("genProtonTable", desc);
}

DEFINE_FWK_MODULE(GenProtonTableProducer);
