/* \class GenParticleDecaySelector
 *
 * \author Luca Lista, INFN
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class GenParticleDecaySelector : public edm::stream::EDProducer<> {
public:
  /// constructor
  GenParticleDecaySelector(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /// process one event
  void produce(edm::Event& e, const edm::EventSetup&) override;
  bool firstEvent_;
  /// source collection name
  edm::EDGetTokenT<reco::GenParticleCollection> srcToken_;
  /// particle type
  edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> tableToken_;
  PdtEntry particle_;
  /// particle status
  int status_;
  /// recursively add a new particle to the output collection
  std::pair<reco::GenParticleRef, reco::GenParticle*> add(reco::GenParticleCollection&,
                                                          const reco::GenParticle&,
                                                          reco::GenParticleRefProd);
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace edm;
using namespace reco;
using namespace std;

GenParticleDecaySelector::GenParticleDecaySelector(const edm::ParameterSet& cfg)
    : firstEvent_(true),
      srcToken_(consumes<GenParticleCollection>(cfg.getParameter<InputTag>("src"))),
      tableToken_(esConsumes()),
      particle_(cfg.getParameter<PdtEntry>("particle")),
      status_(cfg.getParameter<int>("status")) {
  produces<GenParticleCollection>();
}

void GenParticleDecaySelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  desc.addNode(edm::ParameterDescription<int>("particle", true) xor
               edm::ParameterDescription<std::string>("particle", true))
      ->setComment("the PdtEntry can be specified as either an 'int' or via its name using a 'string'");
  desc.add<int>("status");

  descriptions.addDefault(desc);
}

void GenParticleDecaySelector::produce(edm::Event& evt, const edm::EventSetup& es) {
  if (firstEvent_) {
    auto const& pdt = es.getData(tableToken_);
    particle_.setup(pdt);
    firstEvent_ = false;
  }

  Handle<GenParticleCollection> genParticles;
  evt.getByToken(srcToken_, genParticles);
  auto decay = std::make_unique<GenParticleCollection>();
  const GenParticleRefProd ref = evt.getRefBeforePut<GenParticleCollection>();
  for (GenParticleCollection::const_iterator g = genParticles->begin(); g != genParticles->end(); ++g)
    if (g->pdgId() == particle_.pdgId() && g->status() == status_)
      add(*decay, *g, ref);
  evt.put(std::move(decay));
}

pair<GenParticleRef, GenParticle*> GenParticleDecaySelector::add(GenParticleCollection& decay,
                                                                 const GenParticle& p,
                                                                 GenParticleRefProd ref) {
  size_t idx = decay.size();
  GenParticleRef r(ref, idx);
  decay.resize(idx + 1);
  const LeafCandidate& part = p;
  GenParticle g(part);
  size_t n = p.numberOfDaughters();
  for (size_t i = 0; i < n; ++i) {
    pair<GenParticleRef, GenParticle*> d = add(decay, *p.daughterRef(i), ref);
    d.second->addMother(r);
    g.addDaughter(d.first);
  }
  GenParticle& gp = decay[idx] = g;
  return make_pair(r, &gp);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenParticleDecaySelector);
