#include "PhysicsTools/JetMCAlgos/plugins/TauGenJetProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "PhysicsTools/HepMCCandAlgos/interface/GenParticlesHelper.h"

using namespace std;
using namespace edm;
using namespace reco;

TauGenJetProducer::TauGenJetProducer(const edm::ParameterSet& iConfig)
    : inputTagGenParticles_(iConfig.getParameter<InputTag>("GenParticles")),
      tokenGenParticles_(consumes<GenParticleCollection>(inputTagGenParticles_)),
      includeNeutrinos_(iConfig.getParameter<bool>("includeNeutrinos")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)) {
  produces<GenJetCollection>();
}

TauGenJetProducer::~TauGenJetProducer() {}

void TauGenJetProducer::produce(edm::StreamID, Event& iEvent, const EventSetup& iSetup) const {
  Handle<GenParticleCollection> genParticles;

  bool found = iEvent.getByToken(tokenGenParticles_, genParticles);

  if (!found) {
    std::ostringstream err;
    err << " cannot get collection: " << inputTagGenParticles_ << std::endl;
    edm::LogError("TauGenJetProducer") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  auto pOutVisTaus = std::make_unique<GenJetCollection>();

  using namespace GenParticlesHelper;

  GenParticleRefVector allStatus2Taus;
  findParticles(*genParticles, allStatus2Taus, 15, 2);

  for (auto&& allStatus2Tau : allStatus2Taus) {
    // look for all status 1 (stable) descendents
    GenParticleRefVector descendents;
    findDescendents(allStatus2Tau, descendents, 1);

    // CV: skip status 2 taus that radiate-off a photon
    //    --> have a status 2 tau lepton in the list of descendents
    GenParticleRefVector status2TauDaughters;
    findDescendents(allStatus2Tau, status2TauDaughters, 2, 15);
    if (!status2TauDaughters.empty())
      continue;

    // loop on descendents, and take all except neutrinos
    math::XYZTLorentzVector sumVisMom;
    Particle::Charge charge = 0;
    Jet::Constituents constituents;

    if (verbose_)
      cout << "tau " << (allStatus2Tau) << endl;

    for (auto&& descendent : descendents) {
      int absPdgId = abs((descendent)->pdgId());

      // neutrinos
      if (!includeNeutrinos_) {
        if (absPdgId == 12 || absPdgId == 14 || absPdgId == 16)
          continue;
      }

      if (verbose_)
        cout << "\t" << (descendent) << endl;

      charge += (descendent)->charge();
      sumVisMom += (descendent)->p4();

      // need to convert the vector of reference to the constituents
      // to a vector of pointers to build the genjet
      constituents.push_back(refToPtr(descendent));
    }

    math::XYZPoint vertex;
    GenJet::Specific specific;

    GenJet jet(sumVisMom, vertex, specific, constituents);

    if (charge != (allStatus2Tau)->charge())
      std::cout << " charge of Tau: " << (allStatus2Tau) << " not equal to charge of sum of charge of all descendents. "
                << std::endl;

    jet.setCharge(charge);
    pOutVisTaus->push_back(jet);
  }
  iEvent.put(std::move(pOutVisTaus));
}

DEFINE_FWK_MODULE(TauGenJetProducer);
