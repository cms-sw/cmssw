#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace std;
using namespace edm;
using namespace reco;

class TestGenParticleCandidates : public EDAnalyzer {
private: 
  bool dumpHepMC_;
public:
  explicit TestGenParticleCandidates( const ParameterSet & cfg ) : 
    src_( cfg.getParameter<InputTag>( "src" ) ) {
  }
private:
  void analyze( const Event & evt, const EventSetup&) override {
    Handle<CandidateCollection> gen;
    evt.getByLabel( src_, gen );
    size_t n = gen->size();
    if (n == 0) 
      throw Exception(errors::EventCorruption) 
	<< "No particles in genParticleCandidates\n";
    for(size_t i = 0; i < n; ++ i) {
      const Candidate & p = (*gen)[i];
      size_t nd = p.numberOfDaughters();
      if(nd==0 && p.status()==3)
	  throw Exception(errors::EventCorruption) 
	    << "Particle with no daughters and status " << p.status() 
	    << ", pdgId = " << p.pdgId() << "\n";   
      for(size_t j = 0; j < nd; ++ j ) {
	const Candidate * d = p.daughter(j);
	size_t nm = d->numberOfMothers();
	bool noMother = true;
	for(size_t k = 0; k < nm; ++ k ) {
	  if(d->mother(k)==&p) {
	    noMother = false;
	    break;
	  }
	}
	if(noMother)
	  throw Exception(errors::EventCorruption) 
	    << "Inconsistent mother/daughter relation, pdgId = " << d->pdgId() << "\n";
       }
    }
  }    
  InputTag src_;
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( TestGenParticleCandidates );



