#include "PhysicsTools/HepMCCandAlgos/interface/MCCandMatcher.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
using namespace reco;
using namespace std;

MCCandMatcher::MCCandMatcher( const CandMatchMap & map ) :
  CandMatcherBase( map ) {
  initMaps();
}

MCCandMatcher::~MCCandMatcher() {
}

vector<const Candidate *> MCCandMatcher::getDaughters( const Candidate * c ) const {
  vector<const Candidate *> v;
  v.push_back( c );
  int pdgId = c->pdgId();
  if ( c->status() == 3 ) {
    size_t stableIdenticalDaughters = 0;
    const Candidate * identicalDaughter = 0;
    for( size_t i = 0, n = c->numberOfDaughters(); i < n; ++ i ) {
      const Candidate * d = c->daughter( i );
      if ( pdgId == d->pdgId() && d->status() == 1 ) {
	stableIdenticalDaughters ++;
	identicalDaughter = d;
	if ( stableIdenticalDaughters > 1 ) break;
      }
      if ( stableIdenticalDaughters == 1 ) {
	assert( identicalDaughter != 0 );
	v.push_back( identicalDaughter );
      }
    }
  }
  return v;
}
