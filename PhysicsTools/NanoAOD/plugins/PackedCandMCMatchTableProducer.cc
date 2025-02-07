#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include <DataFormats/Math/interface/deltaR.h>
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include <vector>
#include <iostream>

class PackedCandMCMatchTableProducer : public edm::global::EDProducer<> {
public:
  PackedCandMCMatchTableProducer(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        branchName_(params.getParameter<std::string>("branchName")),
        doc_(params.getParameter<std::string>("docString")),
        src_(consumes<reco::CandidateView>(params.getParameter<edm::InputTag>("src"))),
        genPartsToken_(consumes<edm::View<pat::PackedGenParticle>>(params.getParameter<edm::InputTag>("genparticles"))) {
    produces<nanoaod::FlatTable>();
  }

  ~PackedCandMCMatchTableProducer() override {}

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
    edm::Handle<reco::CandidateView> cands;
    iEvent.getByToken(src_, cands);
    unsigned int ncand = cands->size();

    auto tab = std::make_unique<nanoaod::FlatTable>(ncand, objName_, false, true);

    edm::Handle<edm::View<pat::PackedGenParticle>> genParts;
    iEvent.getByToken(genPartsToken_, genParts);

    std::vector<int> key(ncand, -1), flav(ncand, 0);
    for (unsigned int i = 0; i < ncand; ++i) {
      auto cand = cands->ptrAt(i);

      auto iter = std::find_if(genParts->begin(), genParts->end(), [cand](pat::PackedGenParticle genp) {
        return (genp.charge() == cand->charge()) && (deltaR(genp.eta(), genp.phi(), cand->eta(), cand->phi()) < 0.02) &&
               (abs(genp.pt() - cand->pt()) / cand->pt() < 0.2);
      });
      if (iter != genParts->end()) {
        key[i] = iter - genParts->begin();
      }

    }
    tab->addColumn<int>(branchName_ + "Idx", key, "Index into GenCands list for " + doc_);
    iEvent.put(std::move(tab));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("objName")->setComment("name of the nanoaod::FlatTable to extend with this table");
    desc.add<std::string>("branchName")
        ->setComment(
            "name of the column to write (the final branch in the nanoaod will be <objName>_<branchName>Idx and "
            "<objName>_<branchName>Flav");
    desc.add<std::string>("docString")->setComment("documentation to forward to the output");
    desc.add<edm::InputTag>("src")->setComment(
        "physics object collection for the reconstructed objects (e.g. leptons)");
    desc.addOptional<edm::InputTag>("genparticles")->setComment("Collection of genParticles to be stored.");
    descriptions.add("packedCandMcMatchTable", desc);
  }

protected:
  const std::string objName_, branchName_, doc_;
  const edm::EDGetTokenT<reco::CandidateView> src_;
  edm::EDGetTokenT<edm::View<pat::PackedGenParticle>> genPartsToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PackedCandMCMatchTableProducer);
