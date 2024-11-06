// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

class BTVMCFlavourTableProducer : public edm::stream::EDProducer<> {
public:
  explicit BTVMCFlavourTableProducer(const edm::ParameterSet& iConfig)
      : name_(iConfig.getParameter<std::string>("name")),
        src_(consumes<std::vector<pat::Jet> >(iConfig.getParameter<edm::InputTag>("src"))),
        genParticlesToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genparticles"))) {
    produces<nanoaod::FlatTable>();
  }

  ~BTVMCFlavourTableProducer() override {}
  int jet_flavour(const pat::Jet& jet,
                  const std::vector<reco::GenParticle>& gToBB,
                  const std::vector<reco::GenParticle>& gToCC,
                  const std::vector<reco::GenParticle>& neutrinosLepB,
                  const std::vector<reco::GenParticle>& neutrinosLepB_C,
                  const std::vector<reco::GenParticle>& alltaus,
                  bool usePhysForLightAndUndefined) {
    int hflav = abs(jet.hadronFlavour());
    int pflav = abs(jet.partonFlavour());
    int physflav = 0;
    if (!(jet.genJet())) {
      if (pflav == 0)
        return 999;
      else
        return 1000;
    }
    if (jet.genParton())
      physflav = abs(jet.genParton()->pdgId());
    std::size_t nbs = jet.jetFlavourInfo().getbHadrons().size();
    std::size_t ncs = jet.jetFlavourInfo().getcHadrons().size();

    unsigned int nbFromGSP(0);
    for (const reco::GenParticle& p : gToBB) {
      double dr2(reco::deltaR2(jet, p));
      if (dr2 < jetR_ * jetR_)
        ++nbFromGSP;
    }

    unsigned int ncFromGSP(0);
    for (const reco::GenParticle& p : gToCC) {
      double dr2(reco::deltaR2(jet, p));
      if (dr2 < jetR_ * jetR_)
        ++ncFromGSP;
    }

    //std::cout << " jet pt = " << jet.pt() << " hfl = " << hflav << " pfl = " << pflav << " genpart = " << physflav
    //  << " nbFromGSP = " << nbFromGSP << " ncFromGSP = " << ncFromGSP
    //  << " nBhadrons " << nbs << " nCHadrons " << ncs << std::endl;
    if (hflav == 5) {  //B jet
      if (nbs > 1) {
        if (nbFromGSP > 0)
          return 511;
        else
          return 510;
      } else if (nbs == 1) {
        for (std::vector<reco::GenParticle>::const_iterator it = neutrinosLepB.begin(); it != neutrinosLepB.end();
             ++it) {
          if (reco::deltaR2(it->eta(), it->phi(), jet.eta(), jet.phi()) < 0.4 * 0.4) {
            return 520;
          }
        }
        for (std::vector<reco::GenParticle>::const_iterator it = neutrinosLepB_C.begin(); it != neutrinosLepB_C.end();
             ++it) {
          if (reco::deltaR2(it->eta(), it->phi(), jet.eta(), jet.phi()) < 0.4 * 0.4) {
            return 521;
          }
        }
        return 500;
      } else {
        if (usePhysForLightAndUndefined) {
          if (physflav == 21)
            return 0;
          else if (physflav == 3)
            return 2;
          else if (physflav == 2 || physflav == 1)
            return 1;
          else
            return 1000;
        } else
          return 1000;
      }
    } else if (hflav == 4) {  //C jet
      if (ncs > 1) {
        if (ncFromGSP > 0)
          return 411;
        else
          return 410;
      } else
        return 400;
    } else {                   //not a heavy jet
      if (!alltaus.empty()) {  //check for tau in a simplistic way
        bool ishadrtaucontained = true;
        for (const auto& p : alltaus) {
          size_t ndau = p.numberOfDaughters();
          for (size_t i = 0; i < ndau; i++) {
            const reco::Candidate* dau = p.daughter(i);
            int daupid = std::abs(dau->pdgId());
            if (daupid == 13 || daupid == 11) {
              ishadrtaucontained = false;
              break;
            }
            if (daupid != 12 && daupid != 14 && daupid != 16 && reco::deltaR2(*dau, jet) > jetR_ * jetR_) {
              ishadrtaucontained = false;
              break;
            }
          }
        }
        if (ishadrtaucontained)
          return 600;
      }
      if (std::abs(pflav) == 4 || std::abs(pflav) == 5 || nbs || ncs) {
        if (usePhysForLightAndUndefined) {
          if (physflav == 21)
            return 0;
          else if (physflav == 3)
            return 2;
          else if (physflav == 2 || physflav == 1)
            return 1;
          else
            return 1000;
        } else
          return 1000;
      } else if (usePhysForLightAndUndefined) {
        if (physflav == 21)
          return 0;
        else if (physflav == 3)
          return 2;
        else if (physflav == 2 || physflav == 1)
          return 1;
        else
          return 1000;
      } else {
        if (pflav == 21)
          return 0;
        else if (pflav == 3)
          return 2;
        else if (pflav == 2 || pflav == 1)
          return 1;
        else
          return 1000;
      }
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src")->setComment("input Jet collection");
    desc.add<edm::InputTag>("genparticles")->setComment("input genparticles info collection");
    desc.add<std::string>("name")->setComment("name of the genJet FlatTable we are extending with flavour information");
    descriptions.add("btvMCTable", desc);
  }

private:
  void produce(edm::Event&, edm::EventSetup const&) override;

  std::string name_;

  edm::EDGetTokenT<std::vector<pat::Jet> > src_;
  constexpr static double jetR_ = 0.4;

  constexpr static bool usePhysForLightAndUndefined = false;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;
};

// ------------ method called to produce the data  ------------
void BTVMCFlavourTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto jets = iEvent.getHandle(src_);
  // const auto& jetFlavourInfosProd = iEvent.get(genParticlesToken_);
  edm::Handle<reco::GenParticleCollection> genParticlesHandle;
  iEvent.getByToken(genParticlesToken_, genParticlesHandle);
  std::vector<reco::GenParticle> neutrinosLepB;
  std::vector<reco::GenParticle> neutrinosLepB_C;

  std::vector<reco::GenParticle> gToBB;
  std::vector<reco::GenParticle> gToCC;
  std::vector<reco::GenParticle> alltaus;

  unsigned int nJets = jets->size();

  std::vector<unsigned> jet_FlavSplit(nJets);
  for (const reco::Candidate& genC : *genParticlesHandle) {
    const reco::GenParticle& gen = static_cast<const reco::GenParticle&>(genC);
    if (abs(gen.pdgId()) == 12 || abs(gen.pdgId()) == 14 || abs(gen.pdgId()) == 16) {
      const reco::GenParticle* mother = static_cast<const reco::GenParticle*>(gen.mother());
      if (mother != nullptr) {
        if ((abs(mother->pdgId()) > 500 && abs(mother->pdgId()) < 600) ||
            (abs(mother->pdgId()) > 5000 && abs(mother->pdgId()) < 6000)) {
          neutrinosLepB.emplace_back(gen);
        }
        if ((abs(mother->pdgId()) > 400 && abs(mother->pdgId()) < 500) ||
            (abs(mother->pdgId()) > 4000 && abs(mother->pdgId()) < 5000)) {
          neutrinosLepB_C.emplace_back(gen);
        }
      } else {
        std::cout << "No mother" << std::endl;
      }
    }

    int id(std::abs(gen.pdgId()));
    int status(gen.status());

    if (id == 21 && status >= 21 && status <= 59) {  //// Pythia8 hard scatter, ISR, or FSR
      if (gen.numberOfDaughters() == 2) {
        const reco::Candidate* d0 = gen.daughter(0);
        const reco::Candidate* d1 = gen.daughter(1);
        if (std::abs(d0->pdgId()) == 5 && std::abs(d1->pdgId()) == 5 && d0->pdgId() * d1->pdgId() < 0 &&
            reco::deltaR2(*d0, *d1) < jetR_ * jetR_)
          gToBB.push_back(gen);
        if (std::abs(d0->pdgId()) == 4 && std::abs(d1->pdgId()) == 4 && d0->pdgId() * d1->pdgId() < 0 &&
            reco::deltaR2(*d0, *d1) < jetR_ * jetR_)
          gToCC.push_back(gen);
      }
    }

    if (id == 15 && false) {
      alltaus.push_back(gen);
    }
  }
  for (unsigned i_jet = 0; i_jet < nJets; ++i_jet) {
    // from DeepNTuples
    const auto& jet = jets->at(i_jet);

    jet_FlavSplit[i_jet] =
        jet_flavour(jet, gToBB, gToCC, neutrinosLepB, neutrinosLepB_C, alltaus, usePhysForLightAndUndefined);
  }
  auto newtab = std::make_unique<nanoaod::FlatTable>(nJets, name_, false, true);
  newtab->addColumn<int>("FlavSplit",
                         jet_FlavSplit,
                         "Flavour of the jet, numerical codes: "
                         "isG: 0, "
                         "isUD: 1, "
                         "isS: 2, "
                         "isC: 400, "
                         "isCC: 410, "
                         "isGCC: 411, "
                         "isB: 500, "
                         "isBB: 510, "
                         "isGBB: 511, "
                         "isLeptonicB: 520, "
                         "isLeptonicB_C: 521, "
                         "isTAU: 600, "
                         "isPU: 999,"
                         "isUndefined: 1000. "
                         "May be combined to form coarse labels for tagger training and flavour dependent attacks "
                         "using the loss surface.");
  iEvent.put(std::move(newtab));
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(BTVMCFlavourTableProducer);
