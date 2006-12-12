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
  const Candidate * dau;
  if( c->numberOfDaughters() == 1 && 
      status( * c ) == 3 && 
      pdgId( * c ) == pdgId( * ( dau = c->daughter( 0 ) ) ) )
    v.push_back( dau );
  return v;
}
