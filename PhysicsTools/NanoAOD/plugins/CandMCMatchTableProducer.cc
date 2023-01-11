#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <DataFormats/Math/interface/deltaR.h>
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include <vector>
#include <iostream>

class CandMCMatchTableProducer : public edm::global::EDProducer<> {
public:
  CandMCMatchTableProducer(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        branchName_(params.getParameter<std::string>("branchName")),
        doc_(params.getParameter<std::string>("docString")),
        src_(consumes<reco::CandidateView>(params.getParameter<edm::InputTag>("src"))),
        candMap_(consumes<edm::Association<reco::GenParticleCollection>>(params.getParameter<edm::InputTag>("mcMap"))) {
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
    else if (type == "Other")
      type_ = MOther;
    else
      throw cms::Exception("Configuration", "Unsupported objType '" + type + "'\n");

    switch (type_) {
      case MMuon:
        flavDoc_ =
            "1 = prompt muon (including gamma*->mu mu), 15 = muon from prompt tau, "  // continues below
            "5 = muon from b, 4 = muon from c, 3 = muon from light or unknown, 0 = unmatched";
        break;
      case MElectron:
        flavDoc_ =
            "1 = prompt electron (including gamma*->mu mu), 15 = electron from prompt tau, 22 = prompt photon (likely "
            "conversion), "  // continues below
            "5 = electron from b, 4 = electron from c, 3 = electron from light or unknown, 0 = unmatched";
        break;
      case MPhoton:
        flavDoc_ = "1 = prompt photon, 11 = prompt electron, 0 = unknown or unmatched";
        break;
      case MTau:
        flavDoc_ =
            "1 = prompt electron, 2 = prompt muon, 3 = tau->e decay, 4 = tau->mu decay, 5 = hadronic tau decay, 0 = "
            "unknown or unmatched";
        break;
      case MOther:
        flavDoc_ = "1 = from hard scatter, 0 = unknown or unmatched";
        break;
    }

    if (type_ == MTau) {
      candMapVisTau_ =
          consumes<edm::Association<reco::GenParticleCollection>>(params.getParameter<edm::InputTag>("mcMapVisTau"));
    }

    if (type_ == MElectron) {
      candMapDressedLep_ =
          consumes<edm::Association<reco::GenJetCollection>>(params.getParameter<edm::InputTag>("mcMapDressedLep"));
      mapTauAnc_ = consumes<edm::ValueMap<bool>>(params.getParameter<edm::InputTag>("mapTauAnc"));
      genPartsToken_ = consumes<reco::GenParticleCollection>(params.getParameter<edm::InputTag>("genparticles"));
    }
  }

  ~CandMCMatchTableProducer() override {}

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
    const auto& candProd = iEvent.get(src_);
    auto ncand = candProd.size();

    auto tab = std::make_unique<nanoaod::FlatTable>(ncand, objName_, false, true);

    const auto& mapProd = iEvent.get(candMap_);

    edm::Handle<edm::Association<reco::GenParticleCollection>> mapVisTau;
    if (type_ == MTau) {
      iEvent.getByToken(candMapVisTau_, mapVisTau);
    }

    edm::Handle<edm::Association<reco::GenJetCollection>> mapDressedLep;
    edm::Handle<edm::ValueMap<bool>> mapTauAnc;
    edm::Handle<reco::GenParticleCollection> genParts;
    if (type_ == MElectron) {
      iEvent.getByToken(candMapDressedLep_, mapDressedLep);
      iEvent.getByToken(mapTauAnc_, mapTauAnc);
      iEvent.getByToken(genPartsToken_, genParts);
    }

    std::vector<int16_t> key(ncand, -1);
    std::vector<uint8_t> flav(ncand, 0);
    for (unsigned int i = 0; i < ncand; ++i) {
      //std::cout << "cand #" << i << ": pT = " << cands->ptrAt(i)->pt() << ", eta = " << cands->ptrAt(i)->eta() << ", phi = " << cands->ptrAt(i)->phi() << std::endl;
      const auto& cand = candProd.ptrAt(i);
      reco::GenParticleRef match = mapProd[cand];
      reco::GenParticleRef matchVisTau;
      reco::GenJetRef matchDressedLep;
      bool hasTauAnc = false;
      if (type_ == MTau) {
        matchVisTau = (*mapVisTau)[cand];
      }
      if (type_ == MElectron) {
        matchDressedLep = (*mapDressedLep)[cand];
        if (matchDressedLep.isNonnull()) {
          hasTauAnc = (*mapTauAnc)[matchDressedLep];
        }
      }
      if (match.isNonnull())
        key[i] = match.key();
      else if (matchVisTau.isNonnull())
        key[i] = matchVisTau.key();
      else if (type_ != MElectron)
        continue;  // go ahead with electrons, as those may be matched to a dressed lepton

      switch (type_) {
        case MMuon:
          if (match->isPromptFinalState())
            flav[i] = 1;  // prompt
          else if (match->isDirectPromptTauDecayProductFinalState())
            flav[i] = 15;  // tau
          else
            flav[i] = getParentHadronFlag(match);  // 3 = light, 4 = charm, 5 = b
          break;
        case MElectron:
          if (matchDressedLep.isNonnull()) {
            if (matchDressedLep->pdgId() == 22)
              flav[i] = 22;
            else
              flav[i] = (hasTauAnc) ? 15 : 1;

            float minpt = 0;
            const reco::GenParticle* highestPtConstituent = nullptr;
            for (auto& consti : matchDressedLep->getGenConstituents()) {
              if (abs(consti->pdgId()) != 11)
                continue;
              if (consti->pt() < minpt)
                continue;
              minpt = consti->pt();
              highestPtConstituent = consti;
            }
            if (highestPtConstituent) {
              auto iter =
                  std::find_if(genParts->begin(), genParts->end(), [highestPtConstituent](reco::GenParticle genp) {
                    return (abs(genp.pdgId()) == 11) && (deltaR(genp, *highestPtConstituent) < 0.01) &&
                           (abs(genp.pt() - highestPtConstituent->pt()) / highestPtConstituent->pt() < 0.01);
                  });
              if (iter != genParts->end()) {
                key[i] = iter - genParts->begin();
              }
            }
          } else if (!match.isNonnull())
            flav[i] = 0;
          else if (match->isPromptFinalState())
            flav[i] = (match->pdgId() == 22 ? 22 : 1);  // prompt electron or photon
          else if (match->isDirectPromptTauDecayProductFinalState())
            flav[i] = 15;  // tau
          else
            flav[i] = getParentHadronFlag(match);  // 3 = light, 4 = charm, 5 = b
          break;
        case MPhoton:
          if (match->isPromptFinalState() && match->pdgId() == 22)
            flav[i] = 1;  // prompt photon
          else if ((match->isPromptFinalState() || match->isDirectPromptTauDecayProductFinalState()) &&
                   abs(match->pdgId()) == 11)
            flav[i] = 11;  // prompt electron
          break;
        case MTau:
          // CV: assignment of status codes according to https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorking2016#MC_Matching
          if (match.isNonnull() && match->statusFlags().isPrompt() && abs(match->pdgId()) == 11)
            flav[i] = 1;
          else if (match.isNonnull() && match->statusFlags().isPrompt() && abs(match->pdgId()) == 13)
            flav[i] = 2;
          else if (match.isNonnull() && match->isDirectPromptTauDecayProductFinalState() && abs(match->pdgId()) == 11)
            flav[i] = 3;
          else if (match.isNonnull() && match->isDirectPromptTauDecayProductFinalState() && abs(match->pdgId()) == 13)
            flav[i] = 4;
          else if (matchVisTau.isNonnull())
            flav[i] = 5;
          break;
        default:
          flav[i] = match->statusFlags().fromHardProcess();
      };
    }

    tab->addColumn<int16_t>(branchName_ + "Idx", key, "Index into genParticle list for " + doc_);
    tab->addColumn<uint8_t>(branchName_ + "Flav",
                            flav,
                            "Flavour of genParticle (DressedLeptons for electrons) for " + doc_ + ": " + flavDoc_);

    iEvent.put(std::move(tab));
  }

  static int getParentHadronFlag(const reco::GenParticleRef match) {
    bool has4 = false;
    for (unsigned int im = 0, nm = match->numberOfMothers(); im < nm; ++im) {
      reco::GenParticleRef mom = match->motherRef(im);
      assert(mom.isNonnull() && mom.isAvailable());  // sanity
      if (mom.key() >= match.key())
        continue;  // prevent circular refs
      int id = std::abs(mom->pdgId());
      if (id / 1000 == 5 || id / 100 == 5 || id == 5)
        return 5;
      if (id / 1000 == 4 || id / 100 == 4 || id == 4)
        has4 = true;
      if (mom->status() == 2) {
        id = getParentHadronFlag(mom);
        if (id == 5)
          return 5;
        else if (id == 4)
          has4 = true;
      }
    }
    return has4 ? 4 : 3;
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
    desc.add<edm::InputTag>("mcMap")->setComment(
        "tag to an edm::Association<GenParticleCollection> mapping src to gen, such as the one produced by MCMatcher");
    desc.add<std::string>("objType")->setComment(
        "type of object to match (Muon, Electron, Tau, Photon, Other), taylors what's in t Flav branch");
    desc.addOptional<edm::InputTag>("mcMapVisTau")
        ->setComment("as mcMap, but pointing to the visible gen taus (only if objType == Tau)");
    desc.addOptional<edm::InputTag>("mcMapDressedLep")
        ->setComment("as mcMap, but pointing to gen dressed leptons (only if objType == Electrons)");
    desc.addOptional<edm::InputTag>("mapTauAnc")
        ->setComment("Value map of matched gen electrons containing info on the tau ancestry");
    desc.addOptional<edm::InputTag>("genparticles")->setComment("Collection of genParticles to be stored.");
    descriptions.add("candMcMatchTable", desc);
  }

protected:
  const std::string objName_, branchName_, doc_;
  const edm::EDGetTokenT<reco::CandidateView> src_;
  const edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> candMap_;
  edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> candMapVisTau_;
  edm::EDGetTokenT<edm::Association<reco::GenJetCollection>> candMapDressedLep_;
  edm::EDGetTokenT<edm::ValueMap<bool>> mapTauAnc_;
  edm::EDGetTokenT<reco::GenParticleCollection> genPartsToken_;
  enum MatchType { MMuon, MElectron, MTau, MPhoton, MOther } type_;
  std::string flavDoc_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandMCMatchTableProducer);
