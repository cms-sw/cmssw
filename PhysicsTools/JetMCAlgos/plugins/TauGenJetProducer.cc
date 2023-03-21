#include "PhysicsTools/JetMCAlgos/plugins/TauGenJetProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "PhysicsTools/HepMCCandAlgos/interface/GenParticlesHelper.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "DataFormats/TauReco/interface/PFTau.h"

using namespace std;
using namespace edm;
using namespace reco;

namespace {
  //Map to convert names of decay modes to integer codes
  const std::map<std::string, int> decayModeStringToCodeMap = {{"null", PFTau::kNull},
                                                               {"oneProng0Pi0", PFTau::kOneProng0PiZero},
                                                               {"oneProng1Pi0", PFTau::kOneProng1PiZero},
                                                               {"oneProng2Pi0", PFTau::kOneProng2PiZero},
                                                               {"threeProng0Pi0", PFTau::kThreeProng0PiZero},
                                                               {"threeProng1Pi0", PFTau::kThreeProng1PiZero},
                                                               {"electron", PFTau::kRareDecayMode + 1},
                                                               {"muon", PFTau::kRareDecayMode + 2},
                                                               {"rare", PFTau::kRareDecayMode},
                                                               {"tau", PFTau::kNull - 1}};
}  // namespace

TauGenJetProducer::TauGenJetProducer(const edm::ParameterSet& iConfig)
    : tokenGenParticles_(consumes<GenParticleCollection>(iConfig.getParameter<InputTag>("GenParticles"))),
      includeNeutrinos_(iConfig.getParameter<bool>("includeNeutrinos")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)) {
  produces<GenJetCollection>();
}

TauGenJetProducer::~TauGenJetProducer() {}

void TauGenJetProducer::produce(edm::StreamID, Event& iEvent, const EventSetup& iSetup) const {
  Handle<GenParticleCollection> genParticles = iEvent.getHandle(tokenGenParticles_);

  auto pOutVisTaus = std::make_unique<GenJetCollection>();

  using namespace GenParticlesHelper;

  GenParticleRefVector allStatus2Taus;
  findParticles(*genParticles, allStatus2Taus, 15, 2);

  for (IGR iTau = allStatus2Taus.begin(); iTau != allStatus2Taus.end(); ++iTau) {
    // look for all status 1 (stable) descendents
    GenParticleRefVector descendents;
    findDescendents(*iTau, descendents, 1);
    if (descendents.empty()) {
      edm::LogWarning("NoTauDaughters") << "Tau p4: " << (*iTau)->p4() << " vtx: " << (*iTau)->vertex()
                                        << " has no daughters";

      math::XYZPoint vertex;
      GenJet::Specific specific;
      Jet::Constituents constituents;

      constituents.push_back(refToPtr(*iTau));
      GenJet jet((*iTau)->p4(), vertex, specific, constituents);
      jet.setCharge((*iTau)->charge());
      jet.setStatus(decayModeStringToCodeMap.at("tau"));
      pOutVisTaus->emplace_back(std::move(jet));
      continue;
    }
    // CV: skip status 2 taus that radiate-off a photon
    //    --> have a status 2 tau lepton in the list of descendents
    GenParticleRefVector status2TauDaughters;
    findDescendents(*iTau, status2TauDaughters, 2, 15);
    if (!status2TauDaughters.empty())
      continue;

    // loop on descendents, and take all except neutrinos
    math::XYZTLorentzVector sumVisMom;
    Particle::Charge charge = 0;
    Jet::Constituents constituents;

    if (verbose_)
      cout << "tau " << (*iTau) << endl;

    for (IGR igr = descendents.begin(); igr != descendents.end(); ++igr) {
      int absPdgId = abs((*igr)->pdgId());

      // neutrinos
      if (!includeNeutrinos_) {
        if (absPdgId == 12 || absPdgId == 14 || absPdgId == 16)
          continue;
      }

      if (verbose_)
        cout << "\t" << (*igr) << endl;

      charge += (*igr)->charge();
      sumVisMom += (*igr)->p4();

      // need to convert the vector of reference to the constituents
      // to a vector of pointers to build the genjet
      constituents.push_back(refToPtr(*igr));
    }

    math::XYZPoint vertex;
    GenJet::Specific specific;

    GenJet jet(sumVisMom, vertex, specific, constituents);

    if (charge != (*iTau)->charge())
      edm::LogError("TauGenJetProducer") << " charge of Tau: " << (*iTau)
                                         << " not equal to charge of sum of charge of all descendents.\n"
                                         << " Tau's charge: " << (*iTau)->charge() << " sum: " << charge
                                         << " # descendents: " << constituents.size() << "\n";

    jet.setCharge(charge);
    // determine tau decay mode and set it as jet status
    if (auto search = decayModeStringToCodeMap.find(JetMCTagUtils::genTauDecayMode(jet));
        search != decayModeStringToCodeMap.end())
      jet.setStatus(search->second);
    else
      jet.setStatus(decayModeStringToCodeMap.at("null"));
    pOutVisTaus->push_back(jet);
  }
  iEvent.put(std::move(pOutVisTaus));
}

DEFINE_FWK_MODULE(TauGenJetProducer);
