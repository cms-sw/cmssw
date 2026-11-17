// -*- C++ -*-
//
// Package:     PhysicsTools/PatAlgos
// Class:       PATGenJetProducer
//
// reco::GenJet -> pat::GenJet (built from AOD-level generator jets).
// Mirrors PhysicsTools/PatAlgos/plugins/PATJetProducer.cc (CMSSW_20_1_0_pre3):
//   loop over View<reco::GenJet>, build from refAt(idx), fill, add user data, sort by pt, put.
// Differences to PATJetProducer: no b tagging, no tower/PF embedding, no gen-jet match (genJetFwdRef points to
// the input jet itself), no jet ID, no flavour, no parton-jet match, no tracks, no JEC; new parameters
// packedGenParticles (constituent re-keying as PATGenJetSlimmer), daughterOrder, embedGenCandidates
// (generator particles of the jet, counterpart of embedPFCandidates) with genCandidatesForwardMaps, and
// genJetForwardMap.
//
#include "DataFormats/PatCandidates/interface/GenJet.h"
#include "PhysicsTools/PatAlgos/plugins/PATGenJetPluginHelpers.h"

#include "CommonTools/Utils/interface/PtComparator.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace pat {

  class PATGenJetProducer : public edm::stream::EDProducer<> {
  public:
    explicit PATGenJetProducer(const edm::ParameterSet& iConfig);
    ~PATGenJetProducer() override = default;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    enum class DaughterOrder { pt, key };

    void rekeyDaughters(pat::GenJet& jet, const edm::Association<std::vector<pat::PackedGenParticle>>* gp2pgp) const;
    reco::GenParticleFwdPtrVector genCandidates(
        pat::GenJet const& jet, std::vector<const edm::Association<reco::GenParticleCollection>*> const& maps) const;

    // inputs
    const edm::EDGetTokenT<edm::View<reco::GenJet>> jetsToken_;
    const bool rekey_;
    edm::EDGetTokenT<edm::Association<std::vector<pat::PackedGenParticle>>> gp2pgpToken_;
    DaughterOrder daughterOrder_;
    // keep a null daughter for a constituent without packed partner, exactly as PATGenJetSlimmer (needs "key" order)
    bool keepUnmappedDaughters_;
    // also write an Association<vector<pat::GenJet>> from the input gen jets to the output (e.g. for flavour infos)
    bool produceAssociation_;

    // MC (switched off per event on real data)
    bool addGenPartonMatch_;
    bool embedGenPartonMatch_;
    edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> genPartonToken_;

    // generator particles of the jet and forward links
    bool embedGenCandidates_;
    std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>>> genCandidatesForwardMapTokens_;
    bool forwardGenJet_;
    edm::EDGetTokenT<edm::Association<reco::GenJetCollection>> genJetForwardMapToken_;

    // jet charge
    bool addJetCharge_;
    edm::EDGetTokenT<reco::JetFloatAssociation::Container> jetChargeToken_;

    // generic PAT
    bool addEfficiencies_;
    pat::helper::EfficiencyLoader efficiencyLoader_;
    bool addResolutions_;
    pat::helper::KinResolutionsLoader resolutionLoader_;
    const bool useUserData_;
    pat::PATUserDataHelper<pat::GenJet> userDataHelper_;

    GreaterByPt<pat::GenJet> pTComparator_;

    // constituents without a packed partner seen so far (warnings are limited per stream)
    static constexpr unsigned kMaxUnmappedWarnings = 10;
    mutable unsigned nUnmapped_ = 0;
  };

}  // namespace pat

using namespace pat;

PATGenJetProducer::PATGenJetProducer(const edm::ParameterSet& iConfig)
    : jetsToken_(consumes<edm::View<reco::GenJet>>(iConfig.getParameter<edm::InputTag>("jetSource"))),
      rekey_(!iConfig.getParameter<edm::InputTag>("packedGenParticles").label().empty()),
      useUserData_(iConfig.exists("userData")) {
  // ---- constituents
  if (rekey_)
    gp2pgpToken_ = consumes<edm::Association<std::vector<pat::PackedGenParticle>>>(
        iConfig.getParameter<edm::InputTag>("packedGenParticles"));
  const std::string order = iConfig.getParameter<std::string>("daughterOrder");
  if (order == "pt")
    daughterOrder_ = DaughterOrder::pt;
  else if (order == "key")
    daughterOrder_ = DaughterOrder::key;
  else
    throw edm::Exception(edm::errors::Configuration)
        << "PATGenJetProducer: daughterOrder must be \"pt\" or \"key\", got \"" << order << "\"";
  keepUnmappedDaughters_ = iConfig.getParameter<bool>("keepUnmappedDaughters");
  if (keepUnmappedDaughters_ && daughterOrder_ != DaughterOrder::key)
    throw edm::Exception(edm::errors::Configuration)
        << "PATGenJetProducer: keepUnmappedDaughters reproduces PATGenJetSlimmer and needs daughterOrder \"key\"";
  produceAssociation_ = iConfig.getParameter<bool>("produceAssociation");

  // ---- parton match (as PATJetProducer.cc:128-148)
  addGenPartonMatch_ = iConfig.getParameter<bool>("addGenPartonMatch");
  embedGenPartonMatch_ = iConfig.getParameter<bool>("embedGenPartonMatch");
  if (addGenPartonMatch_)
    genPartonToken_ =
        consumes<edm::Association<reco::GenParticleCollection>>(iConfig.getParameter<edm::InputTag>("genPartonMatch"));

  // ---- generator particles of the jet; forward Ptrs through the chain of maps (empty chain: forward = back)
  embedGenCandidates_ = iConfig.getParameter<bool>("embedGenCandidates");
  if (embedGenCandidates_)
    genCandidatesForwardMapTokens_ = edm::vector_transform(
        iConfig.getParameter<std::vector<edm::InputTag>>("genCandidatesForwardMaps"),
        [this](edm::InputTag const& tag) { return consumes<edm::Association<reco::GenParticleCollection>>(tag); });

  // ---- forward ref of genJetFwdRef (empty label: forward = back = the input jet)
  const auto genJetForwardMap = iConfig.getParameter<edm::InputTag>("genJetForwardMap");
  forwardGenJet_ = !genJetForwardMap.label().empty();
  if (forwardGenJet_)
    genJetForwardMapToken_ = consumes<edm::Association<reco::GenJetCollection>>(genJetForwardMap);

  // ---- jet charge
  addJetCharge_ = iConfig.getParameter<bool>("addJetCharge");
  if (addJetCharge_)
    jetChargeToken_ =
        consumes<reco::JetFloatAssociation::Container>(iConfig.getParameter<edm::InputTag>("jetChargeSource"));

  // ---- generic PAT
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_)
    efficiencyLoader_ =
        pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  addResolutions_ = iConfig.getParameter<bool>("addResolutions");
  if (addResolutions_)
    resolutionLoader_ =
        pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"), consumesCollector());
  if (useUserData_)
    userDataHelper_ =
        PATUserDataHelper<pat::GenJet>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());

  produces<std::vector<pat::GenJet>>();
  if (produceAssociation_)
    produces<edm::Association<std::vector<pat::GenJet>>>();
}

/// constituents mapped through the genParticles -> packedGenParticles association
/// (PATGenJetSlimmer.cc:111). "pt": input order kept (AOD gen jets are pt-ordered), later duplicates
/// dropped; "key": packed-key order, as PATGenJetSlimmer (map iteration, PATGenJetSlimmer.cc:114).
/// Without re-keying the original daughters are kept, ordered the same way.
void PATGenJetProducer::rekeyDaughters(pat::GenJet& jet,
                                       const edm::Association<std::vector<pat::PackedGenParticle>>* gp2pgp) const {
  const reco::CompositePtrCandidate::daughters old = jet.daughterPtrVector();
  if (gp2pgp && keepUnmappedDaughters_) {
    // exactly PATGenJetSlimmer.cc:104-116: map by packed key (a missing partner gives one null Ptr, last)
    std::map<unsigned int, reco::CandidatePtr> ptrs;
    for (auto const& dau : old) {
      auto const ref = (*gp2pgp)[dau];
      ptrs[ref.key()] = edm::refToPtr(ref);
    }
    jet.clearDaughters();
    for (auto const& kv : ptrs)
      jet.addDaughter(kv.second);
    return;
  }
  std::vector<reco::CandidatePtr> mapped;
  mapped.reserve(old.size());
  for (auto const& dau : old) {
    if (gp2pgp) {
      auto const ref = (*gp2pgp)[dau];
      if (ref.isNonnull()) {
        mapped.push_back(edm::refToPtr(ref));
      } else {
        // a null mapping is dropped (PATGenJetSlimmer would add a null Ptr); nConstituents can then differ from
        // slimmedGenJets. Reported for the first few cases of each stream, to keep the log small.
        if (nUnmapped_++ < kMaxUnmappedWarnings)
          edm::LogWarning("PATGenJetProducer")
              << "constituent (product " << dau.id() << ", key " << dau.key() << ") of a jet with pt " << jet.pt()
              << " has no packed partner in the packedGenParticles association: dropped from the daughters"
              << (nUnmapped_ == kMaxUnmappedWarnings ? " (further occurrences of this warning are not printed)" : "");
      }
    } else {
      mapped.push_back(dau);
    }
  }
  std::vector<reco::CandidatePtr> ordered;
  ordered.reserve(mapped.size());
  if (daughterOrder_ == DaughterOrder::key) {
    std::map<unsigned int, reco::CandidatePtr> byKey;
    for (auto const& p : mapped)
      byKey.emplace(p.key(), p);  // first wins; same key = same packed particle
    for (auto const& kv : byKey)
      ordered.push_back(kv.second);
  } else {
    std::set<std::pair<edm::ProductID, unsigned int>> seen;
    for (auto const& p : mapped)
      if (seen.emplace(p.id(), p.key()).second)
        ordered.push_back(p);
  }
  jet.clearDaughters();
  for (auto const& p : ordered)
    jet.addDaughter(p);
}

/// the constituents of the jet as generator particles (back Ptr), each forwarded through the chain of maps;
/// a particle without partner keeps forward = back (FwdPtr needs both null or both non-null)
reco::GenParticleFwdPtrVector PATGenJetProducer::genCandidates(
    pat::GenJet const& jet, std::vector<const edm::Association<reco::GenParticleCollection>*> const& maps) const {
  reco::GenParticleFwdPtrVector out;
  out.reserve(jet.numberOfDaughters());
  for (auto const& dau : jet.daughterPtrVector()) {
    const reco::GenParticlePtr back(dau);  // dynamic_cast; null pointer if the constituent is no reco::GenParticle
    if (back.get() == nullptr)
      continue;
    reco::GenParticlePtr fwd = back;
    for (auto const* map : maps) {
      const reco::GenParticleRef ref = (*map)[fwd];
      if (ref.isNull()) {
        fwd = back;
        break;
      }
      fwd = edm::refToPtr(ref);
    }
    out.emplace_back(fwd, back);
  }
  return out;
}

void PATGenJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // PATJetProducer.cc:240: MC inputs off on real data
  const bool addGenPartonMatch = addGenPartonMatch_ && !iEvent.isRealData();

  edm::Handle<edm::View<reco::GenJet>> jets;
  iEvent.getByToken(jetsToken_, jets);

  if (efficiencyLoader_.enabled())
    efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled())
    resolutionLoader_.newEvent(iEvent, iSetup);

  const edm::Association<std::vector<pat::PackedGenParticle>>* gp2pgp = nullptr;
  if (rekey_)
    gp2pgp = &iEvent.get(gp2pgpToken_);

  edm::Handle<edm::Association<reco::GenParticleCollection>> partonMatch;
  if (addGenPartonMatch)
    iEvent.getByToken(genPartonToken_, partonMatch);

  std::vector<const edm::Association<reco::GenParticleCollection>*> genCandidatesForwardMaps;
  genCandidatesForwardMaps.reserve(genCandidatesForwardMapTokens_.size());
  for (auto const& token : genCandidatesForwardMapTokens_)
    genCandidatesForwardMaps.push_back(&iEvent.get(token));

  const edm::Association<reco::GenJetCollection>* genJetForwardMap = nullptr;
  if (forwardGenJet_)
    genJetForwardMap = &iEvent.get(genJetForwardMapToken_);

  edm::Handle<reco::JetFloatAssociation::Container> hJetChargeAss;
  if (addJetCharge_)
    iEvent.getByToken(jetChargeToken_, hJetChargeAss);

  auto patJets = std::make_unique<std::vector<pat::GenJet>>();
  patJets->reserve(jets->size());

  for (size_t idx = 0, n = jets->size(); idx < n; ++idx) {
    const edm::RefToBase<reco::GenJet> genJetRef = jets->refAt(idx);
    // key for the maps keyed to reco::Jet (flavour, JetFloatAssociation): same product ID and key
    const edm::Ptr<reco::Jet> jetPtr(jets->ptrAt(idx));
    pat::GenJet ajet(genJetRef);  // genJetFwdRef: forward and back ref to the input jet

    if (genJetForwardMap) {
      // a jet without partner keeps forward = back (FwdRef needs both null or both non-null)
      const reco::GenJetRef fwd = (*genJetForwardMap)[genJetRef];
      if (fwd.isNonnull())
        ajet.updateFwdGenJetFwdRef(fwd);
    }

    // before the re-keying: the constituents of the input jet are the generator particles
    if (embedGenCandidates_)
      ajet.setGenCandidates(genCandidates(ajet, genCandidatesForwardMaps));

    rekeyDaughters(ajet, gp2pgp);

    if (addGenPartonMatch) {
      reco::GenParticleRef parton = (*partonMatch)[genJetRef];
      if (parton.isNonnull() && parton.isAvailable())
        ajet.setGenParton(parton, embedGenPartonMatch_);
    }

    if (efficiencyLoader_.enabled())
      efficiencyLoader_.setEfficiencies(ajet, genJetRef);
    if (resolutionLoader_.enabled())
      resolutionLoader_.setResolutions(ajet);

    if (addJetCharge_)
      ajet.setJetCharge((*hJetChargeAss)[jetPtr]);

    if (useUserData_)
      userDataHelper_.add(ajet, iEvent, iSetup);

    patJets->push_back(std::move(ajet));
  }

  std::sort(patJets->begin(), patJets->end(), pTComparator_);

  edm::OrphanHandle<std::vector<pat::GenJet>> oh = iEvent.put(std::move(patJets));

  if (produceAssociation_) {
    // input jet -> output jet, through each output jet's back link to its input jet (the output is pt-sorted)
    std::vector<int> mapping(jets->size(), -1);
    for (unsigned int i = 0, n = oh->size(); i < n; ++i) {
      auto const& back = (*oh)[i].genJetFwdRef().backRef();
      if (back.isNonnull() && back.id() == jets.id() && back.key() < mapping.size())
        mapping[back.key()] = i;
    }
    auto asso = std::make_unique<edm::Association<std::vector<pat::GenJet>>>(oh);
    edm::Association<std::vector<pat::GenJet>>::Filler filler(*asso);
    filler.insert(jets, mapping.begin(), mapping.end());
    filler.fill();
    iEvent.put(std::move(asso));
  }
}

void PATGenJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Parameter set of PATJetProducer::fillDescriptions (PATJetProducer.cc:544-614).
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("pat::GenJet producer");

  iDesc.add<edm::InputTag>("jetSource", edm::InputTag("no default"))->setComment("input View<reco::GenJet>");
  iDesc.add<edm::InputTag>("packedGenParticles", edm::InputTag("packedGenParticles"))
      ->setComment(
          "Association<vector<PackedGenParticle>> keyed to the constituents' collection; empty label = no re-keying");
  iDesc
      .ifValue(edm::ParameterDescription<std::string>("daughterOrder", "pt", true),
               edm::allowedValues<std::string>("pt", "key"))
      ->setComment("\"pt\": input (pt) order, duplicates dropped; \"key\": packed-key order as PATGenJetSlimmer");
  iDesc.add<bool>("keepUnmappedDaughters", false)
      ->setComment(
          "keep a null daughter for a constituent without packed partner, as PATGenJetSlimmer (needs \"key\")");
  iDesc.add<bool>("produceAssociation", false)
      ->setComment("also write Association<vector<pat::GenJet>> from the input gen jets to the output");

  iDesc.add<bool>("addGenPartonMatch", false)->setComment("add MC matching");
  iDesc.add<bool>("embedGenPartonMatch", false)->setComment("embed MC matched MC information");
  iDesc.add<edm::InputTag>("genPartonMatch", edm::InputTag())->setComment("input with MC match information");

  iDesc.add<bool>("addJetCharge", false);
  iDesc.add<edm::InputTag>("jetChargeSource", edm::InputTag());

  iDesc.add<bool>("embedGenCandidates", false)
      ->setComment("store the generator particles of the jet (genCandidates), as embedPFCandidates of PATJetProducer");
  iDesc.add<std::vector<edm::InputTag>>("genCandidatesForwardMaps", std::vector<edm::InputTag>())
      ->setComment(
          "chain of Association<GenParticleCollection>, applied in order, from the constituents' collection to the "
          "collection the forward Ptrs point to (no partner: forward = back); empty: forward = back");
  iDesc.add<edm::InputTag>("genJetForwardMap", edm::InputTag())
      ->setComment(
          "Association<GenJetCollection> from the input jets to the collection the forward ref of genJetFwdRef points "
          "to (no partner: forward = back); empty label: forward = back = the input jet");

  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything();  // as PATJetProducer
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<pat::GenJet>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  descriptions.add("patGenJetProducer", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATGenJetProducer);
