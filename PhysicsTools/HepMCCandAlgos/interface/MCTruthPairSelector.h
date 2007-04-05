#ifndef HepMCCandAlgos_MCTruthPairSelector_h
#define HepMCCandAlgos_MCTruthPairSelector_h
/* \class MCTruthPairSelector
 *
 * \author Luca Lista, INFN
 *
 */

#include <set>
#include "DataFormats/Candidate/interface/Candidate.h"

namespace helpers {
  template<typename T>
  struct MCTruthPairSelector {
    MCTruthPairSelector() { }
    template<typename I>
    MCTruthPairSelector( const I & begin, const I & end ) {
      for( I i = begin; i != end; ++i )
	matchIds_.insert( abs( * i ) );
    }
    bool operator()( const T & c, const reco::Candidate & mc ) const {
      if ( mc.status() != 1 ) return false;
      if ( c.charge() != mc.charge() ) return false;
      if ( matchIds_.size() == 0 ) return true;
      return matchIds_.find( abs( mc.pdgId() ) ) != matchIds_.end();
    }
  private:
    std::set<int> matchIds_;
  };
}

#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include <algorithm>
#include <string>
#include <vector>

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<helpers::MCTruthPairSelector<T> > {
      static helpers::MCTruthPairSelector<T> make( const edm::ParameterSet & cfg ) {
	using namespace std;
	const string matchPDGId( "matchPDGId" );
	typedef vector<int> vint;
	vector<string> ints = cfg.template getParameterNamesForType<vint>();
	bool found = find( ints.begin(), ints.end(), matchPDGId ) != ints.end();
	if ( found ) {
	  vint ids = cfg.template getParameter<vint>( matchPDGId );
	  return helpers::MCTruthPairSelector<T>( ids.begin(), ids.end());
	} else {
	  return helpers::MCTruthPairSelector<T>();
	}
      }
    };   
    
  }
}

#endif
