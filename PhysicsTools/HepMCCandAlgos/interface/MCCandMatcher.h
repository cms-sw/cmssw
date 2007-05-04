#ifndef HepMCCandAlgos_MCCandMatcher_h
#define HepMCCandAlgos_MCCandMatcher_h
/* \class MCCandMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/CandUtils/interface/CandMatcher.h"

template<typename C>
class MCCandMatcher : public CandMatcherBase<C> {
public:
  /// constructor
  explicit MCCandMatcher( const typename CandMatcherBase<C>::map_vector & maps );
  /// destructor
  virtual ~MCCandMatcher();
private:
  /// get ultimate daughter skipping status = 3
  virtual std::vector<const reco::Candidate *> getDaughters( const reco::Candidate * ) const;
};

#include "PhysicsTools/HepMCCandAlgos/interface/MCCandMatcher.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

template<typename C>
MCCandMatcher<C>::MCCandMatcher( const typename CandMatcherBase<C>::map_vector & maps ) :
  CandMatcherBase<C>( maps ) {
  CandMatcherBase<C>::initMaps( maps );
}

template<typename C>
MCCandMatcher<C>::~MCCandMatcher() {
}

template<typename C>
std::vector<const reco::Candidate *> MCCandMatcher<C>::getDaughters( const reco::Candidate * c ) const {
  using namespace std;
  using namespace reco;
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

#endif
