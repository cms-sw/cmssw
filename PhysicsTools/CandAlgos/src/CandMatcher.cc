#include "PhysicsTools/CandAlgos/interface/CandMatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include <vector>

using namespace edm;
using namespace std;
using namespace reco;
using namespace reco::modules;

namespace reco {
  namespace helper {
    typedef pair<size_t, double> MatchPair;
    
    struct SortBySecond {
      bool operator()( const MatchPair & p1, const MatchPair & p2 ) const {
	return p1.second < p2.second;
      } 
    };
  }
}

CandMatcherBase::CandMatcherBase( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ),
  matched_( cfg.getParameter<InputTag>( "matched" ) ), 
  distMin_( cfg.getParameter<double>( "distMin" ) ) {
  produces<CandMatchMap>();
}

CandMatcherBase::~CandMatcherBase() {
}

void CandMatcherBase::produce( Event& evt, const EventSetup& ) {
  Handle<CandidateCollection> matched;  
  evt.getByLabel( matched_, matched ) ;
  Handle<CandidateCollection> cands;  
  evt.getByLabel( src_, cands ) ;

  auto_ptr<CandMatchMap> matchMap( new CandMatchMap );
  for( size_t c = 0; c != cands->size(); ++ c ) {
    const Candidate & cand = (*cands)[ c ];
    vector<helper::MatchPair> v;
    for( size_t m = 0; m != matched->size(); ++ m ) {
      const Candidate & match = (*matched)[ m ];
      if ( select( cand, match ) ) {
	double dist = matchDistance( cand, match );
	if ( dist < distMin_ ) v.push_back( make_pair( m, dist ) );
      }
    }
    if ( v.size() > 0 ) {
      size_t mMin = min_element( v.begin(), v.end(), helper::SortBySecond() )->first;
      matchMap->insert( CandidateRef( cands, c ), CandidateRef( matched, mMin ) );
    }
  }

  evt.put( matchMap );
}
