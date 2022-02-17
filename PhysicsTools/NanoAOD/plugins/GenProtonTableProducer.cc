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
  const edm::EDGetTokenT<reco::GenParticleCollection> puCandsToken_, puAltCandsToken_;
  const StringCutObjectSelector<reco::Candidate> protonsCut_;
  const std::string table_name_;
  const double tolerance_;
  bool use_alt_coll_{false};  ///< Are we using premix/mix collection name for PU protons?
};

GenProtonTableProducer::GenProtonTableProducer(const edm::ParameterSet& iConfig)
    : prunedCandsToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("srcPruned"))),
      puCandsToken_(mayConsume<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("srcPUProtons"))),
      puAltCandsToken_(mayConsume<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("srcAltPUProtons"))),
      protonsCut_(iConfig.getParameter<std::string>("cut")),
      table_name_(iConfig.getParameter<std::string>("name")),
      tolerance_(iConfig.getParameter<double>("tolerance")) {
  produces<nanoaod::FlatTable>();
}

void GenProtonTableProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  // define the variables
  std::vector<float> pxs, pys, pzs, vzs;
  std::vector<bool> isPUs;
  // first loop over signal protons
  for (const auto& pruned_cand : iEvent.get(prunedCandsToken_)) {
    if (!protonsCut_(pruned_cand))
      continue;
    pxs.emplace_back(pruned_cand.px());
    pys.emplace_back(pruned_cand.py());
    pzs.emplace_back(pruned_cand.pz());
    vzs.emplace_back(pruned_cand.vz());
    isPUs.emplace_back(false);
  }
  // then loop over pruned candidates ; if already in signal protons, discard
  edm::Handle<reco::GenParticleCollection> hPUCands;
  if (use_alt_coll_ || !iEvent.getByToken(puCandsToken_, hPUCands))
    use_alt_coll_ = iEvent.getByToken(puAltCandsToken_, hPUCands);
  for (const auto& pu_cand : *hPUCands) {
    if (!protonsCut_(pu_cand))
      continue;
    bool associated{false};
    for (size_t i = 0; i < pzs.size(); ++i) {
      if (fabs(1. - pxs.at(i) / pu_cand.px()) < tolerance_ && fabs(1. - pys.at(i) / pu_cand.py()) < tolerance_ &&
          fabs(1. - pzs.at(i) / pu_cand.pz()) < tolerance_) {
        associated = true;
        break;
      }
    }
    if (associated)
      continue;
    pxs.emplace_back(pu_cand.px());
    pys.emplace_back(pu_cand.py());
    pzs.emplace_back(pu_cand.pz());
    vzs.emplace_back(pu_cand.vz());
    isPUs.emplace_back(true);
  }

  auto protons_table = std::make_unique<nanoaod::FlatTable>(isPUs.size(), table_name_, false);
  protons_table->addColumn<float>("px", pxs, "proton horizontal momentum", 8);
  protons_table->addColumn<float>("py", pys, "proton vertical momentum", 8);
  protons_table->addColumn<float>("pz", pzs, "proton longitudinal momentum", 8);
  protons_table->addColumn<float>("vz", vzs, "proton vertex longitudinal coordinate", 8);
  protons_table->addColumn<bool>("isPU", isPUs, "pileup proton?");
  iEvent.put(std::move(protons_table));
}

void GenProtonTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcPruned", edm::InputTag("prunedGenParticles"))
      ->setComment("input source for pruned gen-level particle candidates");
  desc.add<edm::InputTag>("srcPUProtons", edm::InputTag("genPUProtons"))
      ->setComment("input source for pileup protons collection");
  desc.add<edm::InputTag>("srcAltPUProtons", edm::InputTag("genPUProtons", "genPUProtons"))
      ->setComment("alternative input source for pileup protons collection (for premix-mix backward compatibility)");
  desc.add<std::string>("cut", "")->setComment("proton kinematic selection");
  desc.add<std::string>("name", "GenProton")->setComment("flat table name");
  desc.add<std::string>("doc", "generator level information on (signal+PU) protons")
      ->setComment("flat table description");
  desc.add<double>("tolerance", 1.e-3)->setComment("relative difference between the signal and pileup protons momenta");
  descriptions.add("genProtonTable", desc);
}

DEFINE_FWK_MODULE(GenProtonTableProducer);
