/* \class GenParticleCandidate2GenParticleProducer
 *
 * \author Luca Lista, INFN
 *
 * Convert HepMC GenEvent format into a collection of type
 * CandidateCollection containing objects of type GenParticle
 *
 * \version $Id: GenParticleCandidate2GenParticleProducer.cc,v 1.6 2007/10/20 12:09:16 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include <vector>
#include <map>
#include <set>

namespace edm { class ParameterSet; }

class GenParticleCandidate2GenParticleProducer : public edm::EDProducer {
 public:
  /// constructor
  GenParticleCandidate2GenParticleProducer(const edm::ParameterSet &);
  /// destructor
  ~GenParticleCandidate2GenParticleProducer();

 private:
  /// module init at begin of job
  void beginJob(const edm::EventSetup &);
  /// process one event
  void produce(edm::Event& e, const edm::EventSetup&);
  /// source collection name  
  edm::InputTag src_;
};

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
using namespace reco;
using namespace std;

GenParticleCandidate2GenParticleProducer::GenParticleCandidate2GenParticleProducer(const ParameterSet & cfg) :
  src_(cfg.getParameter<InputTag>("src")) {
  produces<GenParticleCollection>();
}

GenParticleCandidate2GenParticleProducer::~GenParticleCandidate2GenParticleProducer() { 
}

void GenParticleCandidate2GenParticleProducer::beginJob(const EventSetup & es) {
}

void GenParticleCandidate2GenParticleProducer::produce(Event& evt, const EventSetup& es) {
  Handle<CandidateCollection> genParticleCandidates;
  evt.getByLabel(src_, genParticleCandidates);
  size_t n = genParticleCandidates->size(); 
  auto_ptr<GenParticleCollection> genParticles(new GenParticleCollection());
  genParticles->reserve(n);
  for(CandidateCollection::const_iterator i = genParticleCandidates->begin(); 
      i != genParticleCandidates->end(); ++i) {
    genParticles->push_back(GenParticle(*i));
  } 
  const GenParticleRefProd ref = evt.getRefBeforePut<GenParticleCollection>();
  for(size_t i = 0; i != n; ++i) {
    const GenParticleCandidate * p = dynamic_cast<const GenParticleCandidate *>(&(*genParticleCandidates)[i]);
    if(p == 0) 
      throw edm::Exception( edm::errors::LogicError ) 
	<< "input collection " << src_ << " contains objecst that are not"
	<< " of type reco::GenParticleCandidate\n";
    const GenParticleCandidate & gpc = * p;
    size_t nd = gpc.numberOfDaughters();
    GenParticle & gp = (*genParticles)[i];
    GenParticleRef gpr(ref, i);
    for(size_t d = 0; d != nd; ++d) {
      CandidateRef dau = gpc.daughterRef(d);
      if(dau.isNull())     
	throw edm::Exception( edm::errors::LogicError ) 
	  << "found null reference to daughter\n";
      size_t idx = dau.key();
      gp.addDaughter(GenParticleRef(ref, idx));
      (*genParticles)[idx].addMother(gpr);
    }
  }
  evt.put(genParticles);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenParticleCandidate2GenParticleProducer );

