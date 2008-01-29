#ifndef HepMCCandAlgos_MCCandMatcher_h
#define HepMCCandAlgos_MCCandMatcher_h
/* \class MCCandMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/CandUtils/interface/CandMatcher.h"

template<typename C1, typename C2 = C1>
class MCCandMatcher : public CandMatcherBase<C1, C2> {
public:
  /// constructor
  explicit MCCandMatcher( const typename CandMatcherBase<C1, C2>::map_vector & maps );
  /// constructor
  explicit MCCandMatcher( const typename CandMatcherBase<C1, C2>::map_type & map );
  /// destructor
  virtual ~MCCandMatcher();
private:
  /// get ultimate daughter skipping status = 3
  virtual std::vector<const reco::Candidate *> getDaughters( const reco::Candidate * ) const;
  /// composite candidate preselection
  virtual bool compositePreselect( const reco::Candidate & c, const reco::Candidate & m ) const;
};

#include "PhysicsTools/HepMCCandAlgos/interface/MCCandMatcher.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

template<typename C1, typename C2>
MCCandMatcher<C1, C2>::MCCandMatcher( const typename CandMatcherBase<C1, C2>::map_vector & maps ) :
  CandMatcherBase<C1, C2>( maps ) {
  CandMatcherBase<C1, C2>::initMaps();
}

template<typename C1, typename C2>
MCCandMatcher<C1, C2>::MCCandMatcher( const typename CandMatcherBase<C1, C2>::map_type & map ) :
  CandMatcherBase<C1, C2>( map ) {
  CandMatcherBase<C1, C2>::initMaps();
}

template<typename C1, typename C2>
MCCandMatcher<C1, C2>::~MCCandMatcher() {
}

template<typename C1, typename C2>
std::vector<const reco::Candidate *> MCCandMatcher<C1, C2>::getDaughters( const reco::Candidate * c ) const {
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

template<typename C1, typename C2>
bool MCCandMatcher<C1, C2>::compositePreselect( const reco::Candidate & c, const reco::Candidate & m ) const {
  // IMPORTANT: select a mother if the number 
  // daughters of c is <= the number of daughters
  // of the matched mom. This is needed because
  // status 3 particles decay in more dummy
  // particles (e.g.: Z0(3)->e+(3)e-(3)Z0(2) or similar)
  // This has the effect, for instance, that a decay
  // like: a10 -> pi+ pi- pi0 is matched to
  // the reconstructed decay rho0 -> pi+ pi-
  return( c.numberOfDaughters() <= m.numberOfDaughters() && 
	  c.charge() == m.charge() );
}

#endif
