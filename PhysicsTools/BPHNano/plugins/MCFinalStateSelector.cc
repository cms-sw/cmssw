// original author: RK18 team

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <vector>
#include <iostream>


class MCFinalStateSelector : public edm::global::EDProducer<> {
public:
  MCFinalStateSelector( edm::ParameterSet const & params ) :
    objName_(params.getParameter<std::string>("objName")),
    branchName_(params.getParameter<std::string>("branchName")),
    doc_(params.getParameter<std::string>("docString")),
    src_(consumes<reco::CandidateView>(params.getParameter<edm::InputTag>("src"))),
    candMap_(consumes<edm::Association<reco::GenParticleCollection>>(params.getParameter<edm::InputTag>("mcMap")))
  {
    produces<nanoaod::FlatTable>();
    const std::string & type = params.getParameter<std::string>("objType");
    if (type == "Muon") type_ = MMuon;
    else if (type == "Electron") type_ = MElectron;
    else if (type == "Tau") type_ = MTau;
    else if (type == "Photon") type_ = MPhoton;
    else if (type == "ProbeTracks") type_ = MTrack;
    else if (type == "Kshort") type_ = MKshort;
    else if (type == "Other") type_ = MOther;
    else throw cms::Exception("Configuration", "Unsupported objType '" + type + "'\n");

    std::cout << "type = " << type << std::endl;
    switch (type_) {
    case MMuon: flavDoc_ = "1 = prompt muon (including gamma*->mu mu), 15 = muon from prompt tau, " // continues below
                             "511 = from B0, 521 = from B+/-, 0 = unknown or unmatched"; break;

    case MElectron: flavDoc_ = "1 = prompt electron (including gamma*->mu mu), 15 = electron from prompt tau, " // continues below
                                 "22 = prompt photon (likely conversion), " // continues below
                                 "511 = from B0, 521 = from B+/-, 0 = unknown or unmatched"; break;
    case MPhoton: flavDoc_ = "1 = prompt photon, 13 = prompt electron, 0 = unknown or unmatched"; break;
    case MTau: flavDoc_    = "1 = prompt electron, 2 = prompt muon, 3 = tau->e decay, 4 = tau->mu decay, 5 = hadronic tau decay, 0 = unknown or unmatched"; break;
    case MTrack: flavDoc_  = "1 = prompt, 511 = from B0, 521 = from B+/-, 0 = unknown or unmatched"; break;

    case MOther: flavDoc_  = "1 = from hard scatter, 0 = unknown or unmatched"; break;
    case MKshort: flavDoc_  = "1 = from hard scatter, 0 = unknown or unmatched"; break;
    }

    if ( type_ == MTau ) {
      candMapVisTau_ = consumes<edm::Association<reco::GenParticleCollection>>(params.getParameter<edm::InputTag>("mcMapVisTau"));
    }
  }

  ~MCFinalStateSelector() override {}

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {

    edm::Handle<reco::CandidateView> cands;
    iEvent.getByToken(src_, cands);
    unsigned int ncand = cands->size();

    auto tab  = std::make_unique<nanoaod::FlatTable>(ncand, objName_, false, true);

    edm::Handle<edm::Association<reco::GenParticleCollection>> map;
    iEvent.getByToken(candMap_, map);

    edm::Handle<edm::Association<reco::GenParticleCollection>> mapVisTau;
    if ( type_ == MTau ) {
      iEvent.getByToken(candMapVisTau_, mapVisTau);
    }

    std::vector<int> key(ncand, -1), flav(ncand, 0);
    for (unsigned int i = 0; i < ncand; ++i) {
      //std::cout << "cand #" << i << ": pT = " << cands->ptrAt(i)->pt() << ", eta = " << cands->ptrAt(i)->eta() << ", phi = " << cands->ptrAt(i)->phi() << std::endl;
      reco::GenParticleRef match = (*map)[cands->ptrAt(i)];
      reco::GenParticleRef matchVisTau;
      if ( type_ == MTau ) {
        matchVisTau = (*mapVisTau)[cands->ptrAt(i)];
      }
      if      ( match.isNonnull()       ) key[i] = match.key();
      else if ( matchVisTau.isNonnull() ) key[i] = matchVisTau.key();
      else continue;
      switch (type_) {
      case MMuon:
        if (match->isPromptFinalState()) flav[i] = 1; // prompt
        else flav[i] = getParentHadronFlag(match); // pdgId of mother
        break;
      case MElectron:
        if (match->isPromptFinalState()) flav[i] = (match->pdgId() == 22 ? 22 : 1); // prompt electron or photon
        else flav[i] = getParentHadronFlag(match); // pdgId of mother
        break;
      case MPhoton:
        if (match->isPromptFinalState()) flav[i] = (match->pdgId() == 22 ? 1 : 13); // prompt electron or photon
        break;
      case MTau:
        // CV: assignment of status codes according to https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsToTauTauWorking2016#MC_Matching
        if      ( match.isNonnull() && match->statusFlags().isPrompt()                  && abs(match->pdgId()) == 11 ) flav[i] = 1;
        else if ( match.isNonnull() && match->statusFlags().isPrompt()                  && abs(match->pdgId()) == 13 ) flav[i] = 2;
        else if ( match.isNonnull() && match->isDirectPromptTauDecayProductFinalState() && abs(match->pdgId()) == 11 ) flav[i] = 3;
        else if ( match.isNonnull() && match->isDirectPromptTauDecayProductFinalState() && abs(match->pdgId()) == 13 ) flav[i] = 4;
        else if ( matchVisTau.isNonnull()                                                                            ) flav[i] = 5;
        break;
      case MTrack:
        if (match->isPromptFinalState()) flav[i] = 1; //prompt
        else flav[i] = getParentHadronFlag(match); // pdgId of mother
        break;
      case MKshort:
        if (match->isPromptFinalState()) flav[i] = 1; //prompt
        else flav[i] = getParentHadronFlag(match); // pdgId of mother
        break;
      default:
        flav[i] = match->statusFlags().fromHardProcess();
      };
    }

    tab->addColumn<int>(branchName_ + "Idx",  key, "Index into genParticle list for " + doc_);
    //for(auto ij : flav) std::cout << " flav = " << ij << " " << (uint8_t)ij << std::endl;
    //tab->addColumn<uint8_t>(branchName_+"Flav", flav, "Flavour of genParticle for "+doc_+": "+flavDoc_, nanoaod::FlatTable::UInt8Column);
    tab->addColumn<int>(branchName_ + "Flav", flav, "Flavour of genParticle for " + doc_ + ": " + flavDoc_);

    iEvent.put(std::move(tab));
  }

  static int getParentHadronFlag(const reco::GenParticleRef match) {
    for (unsigned int im = 0, nm = match->numberOfMothers(); im < nm; ++im) {
      reco::GenParticleRef mom = match->motherRef(im);
      assert(mom.isNonnull() && mom.isAvailable()); // sanity
      if (mom.key() >= match.key()) continue; // prevent circular refs
      int id = std::abs(mom->pdgId());
      return id;
    }
    return 0;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("objName")->setComment("name of the nanoaod::FlatTable to extend with this table");
    desc.add<std::string>("branchName")->setComment("name of the column to write (the final branch in the nanoaod will be <objName>_<branchName>Idx and <objName>_<branchName>Flav");
    desc.add<std::string>("docString")->setComment("documentation to forward to the output");
    desc.add<edm::InputTag>("src")->setComment("physics object collection for the reconstructed objects (e.g. leptons)");
    desc.add<edm::InputTag>("mcMap")->setComment("tag to an edm::Association<GenParticleCollection> mapping src to gen, such as the one produced by MCMatcher");
    desc.add<std::string>("objType")->setComment("type of object to match (Muon, Electron, Tau, Photon, Track, Other), taylors what's in t Flav branch");
    desc.addOptional<edm::InputTag>("mcMapVisTau")->setComment("as mcMap, but pointing to the visible gen taus (only if objType == Tau)");
    descriptions.add("mcFinalStatesTable", desc);
  }

protected:
  const std::string objName_, branchName_, doc_;
  const edm::EDGetTokenT<reco::CandidateView> src_;
  const edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> candMap_;
  edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> candMapVisTau_;
  enum MatchType { MMuon, MElectron, MTau, MPhoton, MTrack, MOther, MKshort } type_;
  std::string flavDoc_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MCFinalStateSelector);

