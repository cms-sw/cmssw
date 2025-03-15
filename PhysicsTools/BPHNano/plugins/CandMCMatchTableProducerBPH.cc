/////////////////////// Candidate MC table ///////////////////////////////////
// original author: RK18 team
/////// Takes as inputs genparts and reco objects and embeds the idxs

#include <iostream>
#include <vector>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class CandMCMatchTableProducerBPH : public edm::global::EDProducer<> {
 public:
  CandMCMatchTableProducerBPH(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        objBranchName_(params.getParameter<std::string>("objBranchName")),
        genBranchName_(params.getParameter<std::string>("genBranchName")),
        doc_(params.getParameter<std::string>("docString")),
        recoObjects_(consumes<reco::CandidateView>(
            params.getParameter<edm::InputTag>("recoObjects"))),
        genParts_(consumes<reco::GenParticleCollection>(
            params.getParameter<edm::InputTag>("genParts"))),
        candMap_(consumes<edm::Association<reco::GenParticleCollection>>(
            params.getParameter<edm::InputTag>("mcMap"))) {
    produces<nanoaod::FlatTable>();
    const std::string& type = params.getParameter<std::string>("objType");
    if (type == "Muon")
      type_ = MMuon;
    else if (type == "Electron")
      type_ = MElectron;
    else if (type == "Tau")
      type_ = MTau;
    else if (type == "Photon")
      type_ = MPhoton;
    else if (type == "Track")
      type_ = MTrack;
    else if (type == "Other")
      type_ = MOther;
    else
      throw cms::Exception("Configuration",
                           "Unsupported objType '" + type + "'\n");

    std::cout << "type = " << type << std::endl;
    switch (type_) {
      case MMuon:
        flavDoc_ =
            "1 = prompt muon (including gamma*->mu mu), 15 = muon from prompt "
            "tau, "  // continues below
            "511 = from B0, 521 = from B+/-, 0 = unknown or unmatched";
        break;

      case MElectron:
        flavDoc_ =
            "1 = prompt electron (including gamma*->mu mu), 15 = electron from "
            "prompt tau, "                              // continues below
            "22 = prompt photon (likely conversion), "  // continues below
            "511 = from B0, 521 = from B+/-, 0 = unknown or unmatched";
        break;
      case MPhoton:
        flavDoc_ =
            "1 = prompt photon, 13 = prompt electron, 0 = unknown or unmatched";
        break;
      case MTau:
        flavDoc_ =
            "1 = prompt electron, 2 = prompt muon, 3 = tau->e decay, 4 = "
            "tau->mu decay, 5 = hadronic tau decay, 0 = unknown or unmatched";
        break;
      case MTrack:
        flavDoc_ =
            "1 = prompt, 511 = from B0, 521 = from B+/-, 0 = unknown or "
            "unmatched";
        break;

      case MOther:
        flavDoc_ = "1 = from hard scatter, 0 = unknown or unmatched";
        break;
    }

    if (type_ == MTau) {
      candMapVisTau_ = consumes<edm::Association<reco::GenParticleCollection>>(
          params.getParameter<edm::InputTag>("mcMapVisTau"));
    }
  }

  ~CandMCMatchTableProducerBPH() override {}

  void produce(edm::StreamID id, edm::Event& iEvent,
               const edm::EventSetup& iSetup) const override {
    edm::Handle<reco::CandidateView> cands;
    iEvent.getByToken(recoObjects_, cands);
    unsigned int ncand = cands->size();

    edm::Handle<reco::GenParticleCollection> gen_parts;
    iEvent.getByToken(genParts_, gen_parts);
    unsigned int ngen = gen_parts->size();

    auto tab =
        std::make_unique<nanoaod::FlatTable>(ncand, objName_, false, true);
    auto tab_reverse =
        std::make_unique<nanoaod::FlatTable>(ngen, "GenPart", false, true);

    edm::Handle<edm::Association<reco::GenParticleCollection>> map;
    iEvent.getByToken(candMap_, map);

    edm::Handle<edm::Association<reco::GenParticleCollection>> mapVisTau;
    if (type_ == MTau) {
      iEvent.getByToken(candMapVisTau_, mapVisTau);
    }

    std::vector<int> key(ncand, -1), flav(ncand, 0);
    std::vector<int> genkey(ngen, -1);

    for (unsigned int i = 0; i < ncand; ++i) {
      // std::cout << "cand #" << i << ": pT = " << cands->ptrAt(i)->pt() << ",
      // eta = " << cands->ptrAt(i)->eta() << ", phi = " <<
      // cands->ptrAt(i)->phi() << std::endl;
      reco::GenParticleRef match = (*map)[cands->ptrAt(i)];
      reco::GenParticleRef matchVisTau;
      if (type_ == MTau) {
        matchVisTau = (*mapVisTau)[cands->ptrAt(i)];
      }
      if (match.isNonnull()) {
        key[i] = match.key();
        genkey[match.key()] = i;
      } else if (matchVisTau.isNonnull())
        key[i] = matchVisTau.key();
      else
        continue;
      switch (type_) {
        case MMuon:
          if (match->isPromptFinalState())
            flav[i] = 1;  // prompt
          else
            flav[i] = getParentHadronFlag(match);  // pdgId of mother
          break;
        case MElectron:
          if (match->isPromptFinalState())
            flav[i] =
                (match->pdgId() == 22 ? 22 : 1);  // prompt electron or photon
          else
            flav[i] = getParentHadronFlag(match);  // pdgId of mother
          break;
        case MPhoton:
          if (match->isPromptFinalState())
            flav[i] =
                (match->pdgId() == 22 ? 1 : 13);  // prompt electron or photon
          break;
        case MTau:
          // CV: assignment of status codes according to
          // https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorking2016#MC_Matching
          if (match.isNonnull() && match->statusFlags().isPrompt() &&
              abs(match->pdgId()) == 11)
            flav[i] = 1;
          else if (match.isNonnull() && match->statusFlags().isPrompt() &&
                   abs(match->pdgId()) == 13)
            flav[i] = 2;
          else if (match.isNonnull() &&
                   match->isDirectPromptTauDecayProductFinalState() &&
                   abs(match->pdgId()) == 11)
            flav[i] = 3;
          else if (match.isNonnull() &&
                   match->isDirectPromptTauDecayProductFinalState() &&
                   abs(match->pdgId()) == 13)
            flav[i] = 4;
          else if (matchVisTau.isNonnull())
            flav[i] = 5;
          break;
        case MTrack:
          if (match->isPromptFinalState())
            flav[i] = 1;  // prompt
          else
            flav[i] = getParentHadronFlag(match);  // pdgId of mother
          break;
        default:
          flav[i] = match->statusFlags().fromHardProcess();
      };
    }

    tab->addColumn<int>(objBranchName_ + "Idx", key,
                        "Index into genParticle list for " + doc_);
    tab->addColumn<int>(objBranchName_ + "Flav", flav,
                        "Flavour of genParticle for " + doc_ + ": " + flavDoc_);
    tab_reverse->addColumn<int>(genBranchName_ + "Idx", genkey,
                                "Index into genParticle list for " + doc_);

    iEvent.put(std::move(tab));
    iEvent.put(std::move(tab_reverse));
  }

  static int getParentHadronFlag(const reco::GenParticleRef match) {
    for (unsigned int im = 0, nm = match->numberOfMothers(); im < nm; ++im) {
      reco::GenParticleRef mom = match->motherRef(im);
      assert(mom.isNonnull() && mom.isAvailable());  // sanity
      if (mom.key() >= match.key()) continue;        // prevent circular refs
      int id = std::abs(mom->pdgId());
      return id;
    }
    return 0;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("objName")->setComment(
        "name of the nanoaod::FlatTable to extend with this table");
    desc.add<std::string>("objBranchName")
        ->setComment(
            "name of the column to write (the final branch in the nanoaod will "
            "be <objName>_<branchName>Idx and <objName>_<branchName>Flav");
    desc.add<std::string>("genBranchName")
        ->setComment(
            "name of the column to write (the final branch in the nanoaod will "
            "be <objName>_<branchName>Idx and <objName>_<branchName>Flav");
    desc.add<std::string>("docString")
        ->setComment("documentation to forward to the output");
    desc.add<edm::InputTag>("recoObjects")
        ->setComment(
            "physics object collection for the reconstructed objects (e.g. "
            "leptons)");
    desc.add<edm::InputTag>("genParts")
        ->setComment(
            "physics object collection for the reconstructed objects (e.g. "
            "leptons)");
    desc.add<edm::InputTag>("mcMap")->setComment(
        "tag to an edm::Association<GenParticleCollection> mapping src to gen, "
        "such as the one produced by MCMatcher");
    desc.add<std::string>("objType")->setComment(
        "type of object to match (Muon, Electron, Tau, Photon, Track, Other), "
        "taylors what's in t Flav branch");
    desc.addOptional<edm::InputTag>("mcMapVisTau")
        ->setComment(
            "as mcMap, but pointing to the visible gen taus (only if objType "
            "== Tau)");
    descriptions.add("candMcMatchTable", desc);
  }

 protected:
  const std::string objName_, objBranchName_, genBranchName_, doc_;
  const edm::EDGetTokenT<reco::CandidateView> recoObjects_;
  const edm::EDGetTokenT<reco::GenParticleCollection> genParts_;
  const edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>>
      candMap_;
  edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>>
      candMapVisTau_;
  enum MatchType { MMuon, MElectron, MTau, MPhoton, MTrack, MOther } type_;
  std::string flavDoc_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandMCMatchTableProducerBPH);
