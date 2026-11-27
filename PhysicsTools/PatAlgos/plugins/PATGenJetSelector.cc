// -*- C++ -*-
//
// Package:     PhysicsTools/PatAlgos
// Class:       PATGenJetSelector
//
// Counterpart of PATJetSelector (PhysicsTools/PatAlgos/plugins/PATObjectSelector.cc:26) for pat::GenJet
//:
//   - same selection parameters: cut, cutLoose, nLoose, filter; the loose counting follows
//     PATJetSelector and PATGenJetSlimmer (first nLoose jets passing cutLoose, in input order)
//   - writes an Association old -> new, as PATGenJetSlimmer does
//   - slimming options, all off by default: dropSpecific, dropDaughters, dropJetVars
//     (string cuts as in PATJetSlimmer, default "0") and modifyJets/modifierConfig
//   - clearDaughters (bool, default False): alias of the PATGenJetSlimmer parameter, so that a configuration copied
//     from slimmedGenJets_cfi validates. True removes the constituents of every jet, as
//     dropDaughters = "1". (The slimmer's False means "re-key", which the producer already did.) The slimmer's
//     dropSpecific is a bool, ours a string cut: write dropSpecific = cms.string('1') for True.
//
#include "DataFormats/PatCandidates/interface/GenJet.h"
#include "PhysicsTools/PatAlgos/plugins/PATGenJetPluginHelpers.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"

#include <memory>
#include <string>
#include <vector>

namespace pat {

  class PATGenJetSelector : public edm::stream::EDFilter<> {
  public:
    explicit PATGenJetSelector(edm::ParameterSet const& params)
        : srcToken_(consumes<edm::View<pat::GenJet>>(params.getParameter<edm::InputTag>("src"))),
          filter_(params.getParameter<bool>("filter")),
          nLoose_(params.getParameter<unsigned>("nLoose")),
          selector_(params.getParameter<std::string>("cut")),
          selectorLoose_(params.getParameter<std::string>("cutLoose")),
          dropSpecific_(params.getParameter<std::string>("dropSpecific")),
          dropDaughters_(params.getParameter<std::string>("dropDaughters")),
          clearDaughters_(params.getParameter<bool>("clearDaughters")),
          dropJetVars_(params.getParameter<std::string>("dropJetVars")),
          modifyJets_(params.getParameter<bool>("modifyJets")) {
      if (modifyJets_)
        jetModifier_ = std::make_unique<pat::ObjectModifier<pat::GenJet>>(
            params.getParameter<edm::ParameterSet>("modifierConfig"), consumesCollector());
      produces<std::vector<pat::GenJet>>();
      produces<edm::Association<std::vector<pat::GenJet>>>();
    }

    bool filter(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
      edm::Handle<edm::View<pat::GenJet>> h_jets;
      iEvent.getByToken(srcToken_, h_jets);

      auto patJets = std::make_unique<std::vector<pat::GenJet>>();

      if (modifyJets_) {
        jetModifier_->setEventContent(iSetup);
        jetModifier_->setEvent(iEvent);
      }

      std::vector<int> mapping;
      mapping.reserve(h_jets->size());
      unsigned nl = 0;  // number of loose jets
      for (auto const& ijet : *h_jets) {
        bool selectedLoose = false;
        if (nLoose_ > 0 && nl < nLoose_ && selectorLoose_(ijet)) {
          selectedLoose = true;
          ++nl;
        }
        if (!(selector_(ijet) || selectedLoose)) {
          mapping.push_back(-1);
          continue;
        }
        mapping.push_back(patJets->size());
        patJets->push_back(ijet);
        pat::GenJet& jet = patJets->back();

        // slimming: selection above is on the unmodified input jet, as PATJetSlimmer
        if (modifyJets_)
          jetModifier_->modify(jet);
        if (dropJetVars_(ijet))
          jet.setNPasses(0);  // as PATJetSlimmer.cc:112-116
        if (clearDaughters_ || dropDaughters_(ijet))
          jet.clearDaughters();
        if (dropSpecific_(ijet))
          jet.setSpecific(reco::GenJet::Specific());
      }

      const bool pass = !patJets->empty();
      edm::OrphanHandle<std::vector<pat::GenJet>> oh = iEvent.put(std::move(patJets));

      auto asso = std::make_unique<edm::Association<std::vector<pat::GenJet>>>(oh);
      edm::Association<std::vector<pat::GenJet>>::Filler filler(*asso);
      filler.insert(h_jets, mapping.begin(), mapping.end());
      filler.fill();
      iEvent.put(std::move(asso));

      return filter_ ? pass : true;
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription iDesc;
      iDesc.setComment("pat::GenJet selector, counterpart of PATJetSelector");
      iDesc.add<edm::InputTag>("src", edm::InputTag("no default"))->setComment("input collection");
      iDesc.add<std::string>("cut", "")->setComment("Jet selection.");
      iDesc.add<std::string>("cutLoose", "")->setComment("Loose jet selection. Will keep nLoose loose jets.");
      iDesc.add<bool>("filter", false)->setComment("Filter selection?");
      iDesc.add<unsigned>("nLoose", 0)->setComment("Keep nLoose loose jets that satisfy cutLoose");
      // slimming,  (off by default)
      iDesc.add<std::string>("dropSpecific", "0")->setComment("cut: reset reco::GenJet::Specific");
      iDesc.add<std::string>("dropDaughters", "0")->setComment("cut: remove the constituents");
      iDesc.add<bool>("clearDaughters", false)
          ->setComment("alias of PATGenJetSlimmer::clearDaughters: True removes the constituents of every jet");
      iDesc.add<std::string>("dropJetVars", "0")->setComment("cut: reset jet variables (nPasses), as PATJetSlimmer");
      iDesc.add<bool>("modifyJets", false)->setComment("apply the modifiers of modifierConfig");
      edm::ParameterSetDescription modifierConfig;
      modifierConfig.setAllowAnything();
      iDesc.add<edm::ParameterSetDescription>("modifierConfig", modifierConfig);
      descriptions.add("patGenJetSelector", iDesc);
    }

  private:
    const edm::EDGetTokenT<edm::View<pat::GenJet>> srcToken_;
    const bool filter_;
    const unsigned nLoose_;
    const StringCutObjectSelector<pat::GenJet> selector_;
    const StringCutObjectSelector<pat::GenJet> selectorLoose_;
    const StringCutObjectSelector<pat::GenJet> dropSpecific_;
    const StringCutObjectSelector<pat::GenJet> dropDaughters_;
    const bool clearDaughters_;
    const StringCutObjectSelector<pat::GenJet> dropJetVars_;
    const bool modifyJets_;
    std::unique_ptr<pat::ObjectModifier<pat::GenJet>> jetModifier_;
  };

}  // namespace pat

#include "FWCore/Framework/interface/MakerMacros.h"
using pat::PATGenJetSelector;
DEFINE_FWK_MODULE(PATGenJetSelector);
