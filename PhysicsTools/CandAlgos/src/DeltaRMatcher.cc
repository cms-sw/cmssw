#include "PhysicsTools/CandAlgos/src/DeltaRMatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include <Math/VectorUtil.h>
#include <vector>

using namespace edm;
using namespace std;
using namespace reco;
using namespace ROOT::Math::VectorUtil;

typedef AssociationMap<OneToOne<CandidateCollection, CandidateCollection> > MatchMap;

namespace helper {
  typedef pair<size_t, double> MatchPair;

  struct SortBySecond {
    bool operator()( const MatchPair & p1, const MatchPair & p2 ) const {
      return p1.second < p2.second;
    } 
  };
}

DeltaRMatcher::DeltaRMatcher( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ),
  matched_( cfg.getParameter<InputTag>( "matched" ) ), 
  drMin_( cfg.getParameter<double>( "drMin" ) ) {
  produces<MatchMap>();
}

DeltaRMatcher::~DeltaRMatcher() {
}

void DeltaRMatcher::produce( Event& evt, const EventSetup& ) {
  Handle<CandidateCollection> matched;  
  evt.getByLabel( matched_, matched ) ;
  Handle<CandidateCollection> cands;  
  evt.getByLabel( src_, cands ) ;

  auto_ptr<MatchMap> matchMap( new MatchMap );
  for( size_t c = 0; c != cands->size(); ++ c ) {
    const Candidate & cand = (*cands)[ c ];
    vector<helper::MatchPair> v;
    for( size_t m = 0; m != matched->size(); ++ m ) {
      const Candidate & match = (*matched)[ m ];
      double dR = DeltaR( cand.p4(), match.p4() );
      if ( dR < drMin_ ) v.push_back( make_pair( m, dR ) );
    }
    size_t mMin = min_element( v.begin(), v.end(), helper::SortBySecond() )->first;
    matchMap->insert( CandidateRef( cands, c ), CandidateRef( matched, mMin ) );
  }

  evt.put( matchMap );
}

