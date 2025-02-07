#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <DataFormats/Math/interface/deltaR.h>
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"

#include <vector>
#include <iostream>

class GenCandMotherTableProducer : public edm::global::EDProducer<> {
public:
  GenCandMotherTableProducer(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        branchName_(params.getParameter<std::string>("branchName")),
        src_(consumes<edm::View<pat::PackedGenParticle>>(params.getParameter<edm::InputTag>("src"))),
        candMap_(consumes<edm::Association<reco::GenParticleCollection>>(params.getParameter<edm::InputTag>("mcMap"))) {
    produces<nanoaod::FlatTable>();
  }

  ~GenCandMotherTableProducer() override {}

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
    edm::Handle<edm::View<pat::PackedGenParticle>> cands;
    iEvent.getByToken(src_, cands);
    unsigned int ncand = cands->size();

    auto tab = std::make_unique<nanoaod::FlatTable>(ncand, objName_, false, true);

    edm::Handle<edm::Association<reco::GenParticleCollection>> map;
    iEvent.getByToken(candMap_, map);

    std::vector<int> key(ncand, -1), fromB(ncand, 0), fromC(ncand, 0);
    for (unsigned int i = 0; i < ncand; ++i) {
      reco::GenParticleRef motherRef = cands->at(i).motherRef();
      reco::GenParticleRef match = (*map)[motherRef];

      if (match.isNull())
        continue;

      key[i] = match.key();
      fromB[i] = isFromB(cands->at(i));
      fromC[i] = isFromC(cands->at(i));
    }

    tab->addColumn<int>(branchName_ + "MotherIdx", key, "Mother index into GenPart list");
    tab->addColumn<uint8_t>("isFromB", fromB, "Is from B hadron: no: 0, any: 1, final: 2");
    tab->addColumn<uint8_t>("isFromC", fromC, "Is from C hadron: no: 0, any: 1, final: 2");
    iEvent.put(std::move(tab));
  }

  bool isFinalB(const reco::Candidate &particle) const {
    if (!CandMCTagUtils::hasBottom(particle))
      return false;

    // check if any of the daughters is also a b hadron
    unsigned int npart = particle.numberOfDaughters();

    for (size_t i = 0; i < npart; ++i) {
      if (CandMCTagUtils::hasBottom(*particle.daughter(i)))
        return false;
    }

    return true;
  }

  int isFromB(const reco::Candidate &particle) const {
    int fromB = 0;

    unsigned int npart = particle.numberOfMothers();
    for (size_t i = 0; i < npart; ++i) {
      const reco::Candidate &mom = *particle.mother(i);
      if (CandMCTagUtils::hasBottom(mom)) {
        fromB = isFinalB(mom) ? 2 : 1;
        break;
      } else
        fromB = isFromB(mom);
    }
    return fromB;
  }

  bool isFinalC(const reco::Candidate &particle) const {
    if (!CandMCTagUtils::hasCharm(particle))
      return false;

    // check if any of the daughters is also a c hadron
    unsigned int npart = particle.numberOfDaughters();

    for (size_t i = 0; i < npart; ++i) {
      if (CandMCTagUtils::hasCharm(*particle.daughter(i)))
        return false;
    }

    return true;
  }

  int isFromC(const reco::Candidate &particle) const {
    int fromC = 0;

    unsigned int npart = particle.numberOfMothers();
    for (size_t i = 0; i < npart; ++i) {
      const reco::Candidate &mom = *particle.mother(i);
      if (CandMCTagUtils::hasCharm(mom)) {
        fromC = isFinalC(mom) ? 2 : 1;
        break;
      } else
        fromC = isFromC(mom);
    }
    return fromC;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("objName", "GenCands")
        ->setComment("name of the nanoaod::FlatTable to extend with this table");
    desc.add<std::string>("branchName", "GenPart")
        ->setComment(
            "name of the column to write (the final branch in the nanoaod will be <objName>_<branchName>Idx and "
            "<objName>_<branchName>Flav");
    desc.add<edm::InputTag>("src", edm::InputTag("packedGenParticles"))
        ->setComment("collection of the packedGenParticles, with association to prunedGenParticles");
    desc.add<edm::InputTag>("mcMap", edm::InputTag("finalGenParticles"))
        ->setComment(
            "tag to an edm::Association<GenParticleCollection> mapping prunedGenParticles to finalGenParticles");
    desc.addOptional<edm::InputTag>("genparticles", edm::InputTag("finalGenparticles"))
        ->setComment("Collection of genParticles to be mapped.");
    descriptions.add("genCandMotherTable", desc);
  }

protected:
  const std::string objName_, branchName_, doc_;
  const edm::EDGetTokenT<edm::View<pat::PackedGenParticle>> src_;
  const edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> candMap_;
  edm::EDGetTokenT<reco::GenParticleCollection> genPartsToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GenCandMotherTableProducer);
