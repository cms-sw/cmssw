/**
  \class    pat::PATGenJetToRecoGenJets
  \brief    Rebuilds a std::vector<reco::GenJet> from a pat::GenJet collection (compatibility layer)

  For jobs that read a MiniAOD in which the slimmed gen-jet collections (slimmedGenJets, slimmedGenJetsAK8,
  slimmedGenJetsAK8SoftDropSubJets) were replaced by pat::GenJet collections: an instance labelled like the
  old collection lets every reader that selects the old collection by label run unchanged.
  Each output jet is the reco::GenJet part of the input pat::GenJet (same order). Options:
    dropSpecific: clear the GenJet::Specific block, as PATGenJetSlimmer with dropSpecific=True
                  (slimmedGenJetsAK8SoftDropSubJets)
  Also writes an Association<vector<reco::GenJet>> from the pat::GenJet collection to the output, with which
  GenJetFlavourInfoPreserver re-keys flavour infos to the rebuilt collection.
  A rebuilt collection is a new product: stored references (edm::Ref/Ptr) to the old collection do not point to it.
*/

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/PatCandidates/interface/GenJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>
#include <vector>

namespace pat {

  class PATGenJetToRecoGenJets : public edm::global::EDProducer<> {
  public:
    explicit PATGenJetToRecoGenJets(const edm::ParameterSet& iConfig)
        : srcToken_(consumes<std::vector<pat::GenJet>>(iConfig.getParameter<edm::InputTag>("src"))),
          dropSpecific_(iConfig.getParameter<bool>("dropSpecific")) {
      produces<reco::GenJetCollection>();
      produces<edm::Association<reco::GenJetCollection>>();
    }

    void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
      edm::Handle<std::vector<pat::GenJet>> src;
      iEvent.getByToken(srcToken_, src);

      auto out = std::make_unique<reco::GenJetCollection>();
      out->reserve(src->size());
      for (auto const& jet : *src) {
        out->push_back(static_cast<reco::GenJet const&>(jet));
        if (dropSpecific_)
          out->back().setSpecific(reco::GenJet::Specific());
      }
      const unsigned int n = out->size();
      edm::OrphanHandle<reco::GenJetCollection> oh = iEvent.put(std::move(out));

      // pat::GenJet i -> reco::GenJet i
      std::vector<int> mapping(n);
      for (unsigned int i = 0; i < n; ++i)
        mapping[i] = i;
      auto asso = std::make_unique<edm::Association<reco::GenJetCollection>>(oh);
      edm::Association<reco::GenJetCollection>::Filler filler(*asso);
      filler.insert(src, mapping.begin(), mapping.end());
      filler.fill();
      iEvent.put(std::move(asso));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.setComment(
          "rebuilds a vector<reco::GenJet> (the reco::GenJet part, same order) from a pat::GenJet collection");
      desc.add<edm::InputTag>("src", edm::InputTag("patGenJetsAK4"))->setComment("input vector<pat::GenJet>");
      desc.add<bool>("dropSpecific", false)
          ->setComment("clear GenJet::Specific, as PATGenJetSlimmer with dropSpecific=True");
      descriptions.add("patGenJetToRecoGenJets", desc);
    }

  private:
    const edm::EDGetTokenT<std::vector<pat::GenJet>> srcToken_;
    const bool dropSpecific_;
  };

}  // namespace pat

using pat::PATGenJetToRecoGenJets;
DEFINE_FWK_MODULE(PATGenJetToRecoGenJets);
