// -*- C++ -*-
//
// Package:    HiSignalParticleProducer
// Class:      HiSignalParticleProducer
//
/**\class HiSignalParticleProducer HiSignalParticleProducer.cc yetkin/HiSignalParticleProducer/src/HiSignalParticleProducer.cc
 Description: <one line class summary>
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue Jul 21 04:26:01 EDT 2009
//
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

using namespace std;
using namespace edm;

//
// class decleration
//

class HiSignalParticleProducer : public edm::global::EDProducer<> {
public:
  explicit HiSignalParticleProducer(const edm::ParameterSet&);
  ~HiSignalParticleProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::GenParticle> > genParticleSrc_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HiSignalParticleProducer::HiSignalParticleProducer(const edm::ParameterSet& iConfig)
    : genParticleSrc_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("src"))) {
  std::string alias = (iConfig.getParameter<InputTag>("src")).label();
  produces<reco::GenParticleCollection>().setBranchAlias(alias);
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void HiSignalParticleProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  using namespace edm;
  using namespace reco;

  auto signalGenParticles = std::make_unique<GenParticleCollection>();

  edm::Handle<edm::View<GenParticle> > genParticles;
  iEvent.getByToken(genParticleSrc_, genParticles);

  int genParticleSize = genParticles->size();

  for (int igenParticle = 0; igenParticle < genParticleSize; ++igenParticle) {
    const GenParticle* genParticle = &((*genParticles)[igenParticle]);
    if (genParticle->collisionId() == 0) {
      signalGenParticles->push_back(*genParticle);
    }
  }

  iEvent.put(std::move(signalGenParticles));
}

void HiSignalParticleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Selects genParticles from collision id = 0");
  desc.add<std::string>("src", "genParticles");
  descriptions.add("HiSignalParticleProducer", desc);
}

DEFINE_FWK_MODULE(HiSignalParticleProducer);
