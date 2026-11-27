// -*- C++ -*-
//
// Package:     PhysicsTools/PatAlgos
// Class:       PATGenJetSubstructurePacker
//
// One module doing what BoostedJetMerger + JetSubstructurePacker do for reco AK8 jets
//. Gen groomed jets are reco::BasicJet, so they cannot become pat::GenJet;
// for each algorithm (VPSet "algos"):
//   1. subjet output (instance subjetInstance): for each groomed jet in input order, each daughter
//      (a Ptr into the subjet source) is matched to the pat::GenJet subjet whose originalObjectRef()
//      equals it (BoostedJetMerger's FindCorrectedSubjet), copied in that order.
//   2. fat jets: the first groomed jet within distMax (JetSubstructurePacker.cc:56-70) gives
//      - user floats from the groomedUserFloats value maps (ECFs), label = encoded tag minus postfixes
//      - the subjet Ptrs, pointing into this module's OWN subjet output (getRefBeforePut, as
//        BoostedJetMerger.cc:18), attached with addSubjets(…, label)
//   3. Association old -> new written.
//   4. fixDaughters (default False): replace the constituents covered by a subjet with that subjet,
//      as JetSubstructurePacker.cc:83-111 (no packed-candidate mapping needed: gen constituents are
//      already packed in both the fat jets and the subjets).
//
#include "DataFormats/PatCandidates/interface/GenJet.h"
#include "PhysicsTools/PatAlgos/plugins/PATGenJetPluginHelpers.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/transform.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace pat {

  class PATGenJetSubstructurePacker : public edm::stream::EDProducer<> {
  public:
    explicit PATGenJetSubstructurePacker(const edm::ParameterSet& iConfig);
    ~PATGenJetSubstructurePacker() override = default;

    void produce(edm::Event& iEvent, const edm::EventSetup&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    struct Algo {
      std::string label;
      std::string subjetInstance;
      edm::EDGetTokenT<edm::View<reco::Jet>> groomedToken;
      edm::EDGetTokenT<edm::View<pat::GenJet>> subjetToken;
      std::vector<edm::EDGetTokenT<edm::ValueMap<float>>> userFloatTokens;
      std::vector<std::string> userFloatLabels;
    };

    static void fixDaughters(pat::GenJet& fat,
                             pat::GenJetPtrCollection const& subjets,
                             std::vector<pat::GenJet> const& subjetsOut);

    const edm::EDGetTokenT<edm::View<pat::GenJet>> jetToken_;
    const double distMax_;
    const bool fixDaughters_;
    std::vector<Algo> algos_;
  };

}  // namespace pat

using namespace pat;

PATGenJetSubstructurePacker::PATGenJetSubstructurePacker(const edm::ParameterSet& iConfig)
    : jetToken_(consumes<edm::View<pat::GenJet>>(iConfig.getParameter<edm::InputTag>("jetSrc"))),
      distMax_(iConfig.getParameter<double>("distMax")),
      fixDaughters_(iConfig.getParameter<bool>("fixDaughters")) {
  const auto postfixes = iConfig.getParameter<std::vector<std::string>>("labelPostfixesToStrip");
  std::set<std::string> instances;
  for (auto const& pset : iConfig.getParameter<std::vector<edm::ParameterSet>>("algos")) {
    Algo a;
    a.label = pset.getParameter<std::string>("label");
    a.subjetInstance = pset.getParameter<std::string>("subjetInstance");
    if (a.subjetInstance.empty() || !instances.insert(a.subjetInstance).second)
      throw edm::Exception(edm::errors::Configuration)
          << "PATGenJetSubstructurePacker: subjetInstance must be non-empty and unique per algo, got \""
          << a.subjetInstance << "\"";
    a.groomedToken = consumes<edm::View<reco::Jet>>(pset.getParameter<edm::InputTag>("groomedJets"));
    a.subjetToken = consumes<edm::View<pat::GenJet>>(pset.getParameter<edm::InputTag>("subjets"));
    for (auto const& tag : pset.getParameter<std::vector<edm::InputTag>>("groomedUserFloats")) {
      a.userFloatTokens.push_back(consumes<edm::ValueMap<float>>(tag));
      a.userFloatLabels.push_back(pat::genjethelpers::strippedLabel(tag, postfixes));
    }
    produces<std::vector<pat::GenJet>>(a.subjetInstance);
    algos_.push_back(std::move(a));
  }
  produces<std::vector<pat::GenJet>>();
  produces<edm::Association<std::vector<pat::GenJet>>>();
}

/// 4. fixDaughters, as JetSubstructurePacker.cc:83-111 for the first algorithm. The subjet Ptrs point into
/// the output being built (not yet in the event), so the subjet constituents are read from that vector.
void PATGenJetSubstructurePacker::fixDaughters(pat::GenJet& fat,
                                               pat::GenJetPtrCollection const& subjets,
                                               std::vector<pat::GenJet> const& subjetsOut) {
  const std::vector<reco::CandidatePtr> jdaus = fat.daughterPtrVector();
  std::vector<reco::CandidatePtr> daughtersInSubjets, daughtersNew;
  for (const edm::Ptr<pat::GenJet>& subjet : subjets) {
    const std::vector<reco::CandidatePtr>& sjdaus = subjetsOut[subjet.key()].daughterPtrVector();
    bool skipSubjet = false;
    for (auto const& dau : sjdaus) {
      if (std::find(jdaus.begin(), jdaus.end(), dau) == jdaus.end()) {
        skipSubjet = true;
        break;
      }
    }
    if (skipSubjet)
      continue;
    daughtersInSubjets.insert(daughtersInSubjets.end(), sjdaus.begin(), sjdaus.end());
    daughtersNew.push_back(reco::CandidatePtr(subjet));
  }
  for (auto const& dau : jdaus)
    if (std::find(daughtersInSubjets.begin(), daughtersInSubjets.end(), dau) == daughtersInSubjets.end())
      daughtersNew.push_back(dau);
  fat.clearDaughters();
  for (auto const& dau : daughtersNew)
    fat.addDaughter(dau);
}

void PATGenJetSubstructurePacker::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<edm::View<pat::GenJet>> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);

  auto outputs = std::make_unique<std::vector<pat::GenJet>>();
  outputs->reserve(jetHandle->size());
  for (auto const& jet : *jetHandle)
    outputs->push_back(jet);

  for (auto const& algo : algos_) {
    edm::Handle<edm::View<reco::Jet>> groomed;
    iEvent.getByToken(algo.groomedToken, groomed);
    edm::Handle<edm::View<pat::GenJet>> subjetsIn;
    iEvent.getByToken(algo.subjetToken, subjetsIn);
    std::vector<edm::Handle<edm::ValueMap<float>>> maps(algo.userFloatTokens.size());
    for (size_t k = 0; k < maps.size(); ++k)
      iEvent.getByToken(algo.userFloatTokens[k], maps[k]);

    // ---- 1. subjet output, ordered by groomed jet and groomed daughter (BoostedJetMerger.cc:30-41)
    auto subjetsOut = std::make_unique<std::vector<pat::GenJet>>();
    const edm::RefProd<pat::GenJetCollection> h_subOut =
        iEvent.getRefBeforePut<pat::GenJetCollection>(algo.subjetInstance);
    std::vector<pat::GenJetPtrCollection> subjetPtrsOfGroomed(groomed->size());
    for (size_t g = 0; g < groomed->size(); ++g) {
      reco::Jet const& gj = (*groomed)[g];
      for (size_t d = 0; d < gj.numberOfDaughters(); ++d) {
        const reco::CandidatePtr dau = gj.daughterPtr(d);
        auto found = std::find_if(subjetsIn->begin(), subjetsIn->end(), [&dau](pat::GenJet const& s) {
          return s.originalObjectRef() == dau;
        });
        if (found == subjetsIn->end())
          continue;  // as BoostedJetMerger: unmatched daughters are skipped
        subjetsOut->push_back(*found);
        subjetPtrsOfGroomed[g].emplace_back(h_subOut.id(), subjetsOut->size() - 1, h_subOut.productGetter());
      }
    }

    // ---- 2. fat jets: first groomed jet within distMax (JetSubstructurePacker.cc:53-74)
    for (auto& fat : *outputs) {
      pat::GenJetPtrCollection nextSubjets;
      const double dRMin = distMax_;
      for (size_t g = 0; g < groomed->size(); ++g) {
        if (reco::deltaR(fat, (*groomed)[g]) < dRMin) {
          const edm::Ptr<reco::Jet> gptr = groomed->ptrAt(g);
          for (size_t k = 0; k < maps.size(); ++k)
            fat.addUserFloat(algo.userFloatLabels[k], (*maps[k])[gptr]);
          nextSubjets = subjetPtrsOfGroomed[g];
          break;
        }
      }
      if (fixDaughters_ && &algo == &algos_.front())
        fixDaughters(fat, nextSubjets, *subjetsOut);
      fat.addSubjets(nextSubjets, algo.label);  // empty collection if no match, as JetSubstructurePacker
    }

    iEvent.put(std::move(subjetsOut), algo.subjetInstance);
  }

  // ---- 3. association
  std::vector<int> mapping(jetHandle->size());
  for (size_t i = 0; i < mapping.size(); ++i)
    mapping[i] = i;
  edm::OrphanHandle<std::vector<pat::GenJet>> oh = iEvent.put(std::move(outputs));
  auto asso = std::make_unique<edm::Association<std::vector<pat::GenJet>>>(oh);
  edm::Association<std::vector<pat::GenJet>>::Filler filler(*asso);
  filler.insert(jetHandle, mapping.begin(), mapping.end());
  filler.fill();
  iEvent.put(std::move(asso));
}

void PATGenJetSubstructurePacker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("pat::GenJet substructure packer");
  iDesc.add<edm::InputTag>("jetSrc", edm::InputTag("no default"))->setComment("fat jets, View<pat::GenJet>");
  iDesc.add<double>("distMax", 0.8)->setComment("max deltaR fat jet - groomed jet (first match wins)");
  iDesc.add<bool>("fixDaughters", false)->setComment("replace constituents covered by subjets with the subjets");
  iDesc.add<std::vector<std::string>>("labelPostfixesToStrip", std::vector<std::string>())
      ->setComment("postfixes stripped from the groomedUserFloats module labels (e.g. ForPatGenJets)");
  edm::ParameterSetDescription algo;
  algo.add<std::string>("label", "SoftDropPuppi")->setComment("subjet label on the fat jet (reco name)");
  algo.add<edm::InputTag>("groomedJets", edm::InputTag("ak8GenJetsNoNuSoftDrop"))
      ->setComment("View<reco::Jet>, daughters are Ptrs into the subjet source");
  algo.add<edm::InputTag>("subjets", edm::InputTag("no default"))
      ->setComment("View<pat::GenJet> made by PATGenJetProducer from the subjet source");
  algo.add<std::vector<edm::InputTag>>("groomedUserFloats", std::vector<edm::InputTag>())
      ->setComment("ValueMap<float> keyed to the groomed jets, copied to the matched fat jet");
  algo.add<std::string>("subjetInstance", "SubJets")->setComment("instance of this module's subjet output");
  std::vector<edm::ParameterSet> defaults(1);
  defaults[0].addParameter<std::string>("label", "SoftDropPuppi");
  defaults[0].addParameter<edm::InputTag>("groomedJets", edm::InputTag("ak8GenJetsNoNuSoftDrop"));
  defaults[0].addParameter<edm::InputTag>("subjets", edm::InputTag("patGenJetsAK8SoftDropSubjets"));
  defaults[0].addParameter<std::vector<edm::InputTag>>("groomedUserFloats", std::vector<edm::InputTag>());
  defaults[0].addParameter<std::string>("subjetInstance", "SubJets");
  iDesc.addVPSet("algos", algo, defaults);
  descriptions.add("patGenJetSubstructurePacker", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATGenJetSubstructurePacker);
