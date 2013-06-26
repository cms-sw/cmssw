/* \class GenParticleDecaySelector
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleDecaySelector.cc,v 1.5 2013/02/27 23:16:51 wmtan Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class GenParticleDecaySelector : public edm::EDProducer {
public:
  /// constructor
  GenParticleDecaySelector(const edm::ParameterSet&);
private:
  /// process one event
  void produce(edm::Event& e, const edm::EventSetup&) override;
  bool firstEvent_;
  /// source collection name  
  edm::InputTag src_;  
  /// particle type
  PdtEntry particle_;
  /// particle status
  int status_;
  /// recursively add a new particle to the output collection
  std::pair<reco::GenParticleRef, reco::GenParticle*>
  add(reco::GenParticleCollection&, const reco::GenParticle &, 
      reco::GenParticleRefProd);
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace edm;
using namespace reco;
using namespace std;

GenParticleDecaySelector::GenParticleDecaySelector(const edm::ParameterSet& cfg) :
  firstEvent_(true),
  src_(cfg.getParameter<InputTag>("src")),
  particle_(cfg.getParameter<PdtEntry>("particle")),
  status_(cfg.getParameter<int>("status")) {
  produces<GenParticleCollection>();
}

void GenParticleDecaySelector::produce(edm::Event& evt, const edm::EventSetup& es) {
  if (firstEvent_) {particle_.setup(es); firstEvent_ = false;}

  Handle<GenParticleCollection> genParticles;
  evt.getByLabel(src_, genParticles);
  auto_ptr<GenParticleCollection> decay(new GenParticleCollection);
  const GenParticleRefProd ref = evt.getRefBeforePut<GenParticleCollection>();
  for(GenParticleCollection::const_iterator g = genParticles->begin();
      g != genParticles->end(); ++g)
    if(g->pdgId() == particle_.pdgId() && g->status() == status_)
      add(*decay, *g, ref);
  evt.put(decay);
}

pair<GenParticleRef, GenParticle*> GenParticleDecaySelector::add(GenParticleCollection & decay, const GenParticle & p,
								 GenParticleRefProd ref) {
  size_t idx = decay.size();
  GenParticleRef r(ref, idx);
  decay.resize(idx+1);
  const LeafCandidate & part = p;
  GenParticle g(part);
  size_t n = p.numberOfDaughters();
  for(size_t i = 0; i < n; ++i) {
    pair<GenParticleRef, GenParticle*> d = add(decay, *p.daughterRef(i), ref);
    d.second->addMother(r);
    g.addDaughter(d.first);
  }
  GenParticle & gp = decay[idx] = g;
  return make_pair(r, &gp);
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenParticleDecaySelector);

