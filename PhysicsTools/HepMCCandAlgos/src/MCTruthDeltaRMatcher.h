#ifndef MCTruthDeltaRMatcher_h
#define MCTruthDeltaRMatcher_h
/* \class DeltaRMatcher
 *
 * Producer fo simple match map
 * based on DeltaR
 *
 */
#include "PhysicsTools/CandAlgos/interface/CandMatcher.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include <set>
#include <algorithm>
#include <iterator>

namespace helpers {
  struct MCTruthPairSelector {
    explicit MCTruthPairSelector( const edm::ParameterSet & cfg ) { 
      using namespace std;
      const string matchPDGId( "matchPDGId" );
      typedef vector<int> vint;
      vector<string> ints = cfg.getParameterNamesForType<vint>();
      bool found = find( ints.begin(), ints.end(), matchPDGId ) != ints.end();
      if ( found ) {
	vint ids = cfg.getParameter<vint>( matchPDGId );
	for( vint::const_iterator i = ids.begin(); i != ids.end(); ++i )
	  matchIds_.insert( abs( * i ) );
      }
    }
    bool operator()( const reco::Candidate & c, const reco::Candidate & mc ) const {
      if ( reco::status( mc ) != 1 ) return false;
      if ( c.charge() != mc.charge() ) return false;
      if ( matchIds_.size() == 0 ) return true;
      return matchIds_.find( abs( reco::pdgId( mc ) ) ) != matchIds_.end();
    }
  private:
    std::set<int> matchIds_;
  };
}

typedef reco::modules::CandMatcher<helpers::MCTruthPairSelector> MCTruthDeltaRMatcher;

#endif
