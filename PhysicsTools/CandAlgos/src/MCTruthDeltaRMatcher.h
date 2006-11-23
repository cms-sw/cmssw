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

namespace helpers {
  struct MCTruthPairSelector {
    explicit MCTruthPairSelector( const edm::ParameterSet & ) { }
    bool operator()( const reco::Candidate & c, const reco::Candidate & mc ) const {
      if ( reco::status( mc ) != 1 ) return false;
      if ( c.charge() != mc.charge() ) return false;
      return true;
    }
  };
}

typedef reco::modules::CandMatcher<helpers::MCTruthPairSelector> MCTruthDeltaRMatcher;

#endif
