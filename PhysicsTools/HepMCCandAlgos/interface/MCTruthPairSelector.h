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
    MCTruthPairSelector( bool checkCharge = false ) : 
      checkCharge_( checkCharge ) { }
    template<typename I>
    MCTruthPairSelector( const I & begin, const I & end, bool checkCharge = false ) :
      checkCharge_( checkCharge ) {
      for( I i = begin; i != end; ++i )
	matchIds_.insert( std::abs( * i ) );
    }
    bool operator()( const T & c, const reco::Candidate & mc ) const {
      if ( mc.status() != 1 ) return false;
      if ( checkCharge_ && c.charge() != mc.charge() ) return false;
      if ( matchIds_.size() == 0 ) return true;
      return matchIds_.find( std::abs( mc.pdgId() ) ) != matchIds_.end();
    }
  private:
    std::set<int> matchIds_;
    bool checkCharge_;
  };
}

#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include <algorithm>
#include <string>
#include <vector>

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<helpers::MCTruthPairSelector<T> > {
      static helpers::MCTruthPairSelector<T> make( const edm::ParameterSet & cfg ) {
	const std::string matchPDGId( "matchPDGId" );
	const std::string checkCharge( "checkCharge" );
	bool ck = false;
	std::vector<std::string> bools = cfg.template getParameterNamesForType<bool>();
	bool found = find( bools.begin(), bools.end(), checkCharge ) != bools.end();
	if (found) ck = cfg.template getParameter<bool>( checkCharge ); 
	typedef std::vector<int> vint;
	std::vector<std::string> ints = cfg.template getParameterNamesForType<vint>();
	found = find( ints.begin(), ints.end(), matchPDGId ) != ints.end();
	if ( found ) {
	  vint ids = cfg.template getParameter<vint>( matchPDGId );
	  return helpers::MCTruthPairSelector<T>( ids.begin(), ids.end(), ck );
	} else {
	  return helpers::MCTruthPairSelector<T>( ck );
	}
      }
    };   
    
  }
}

#endif

