#include "PhysicsTools/CandUtils/interface/ThreeBodyCombiner.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
using namespace reco;
using namespace std;

ThreeBodyCombiner::ThreeBodyCombiner( const boost::shared_ptr<CandSelector> & sel, bool ck, const vector<int> & dauCharge, int q ) :
  checkCharge( ck ), dauCharge_(dauCharge), charge( 0 ), overlap(), select( sel ) {
  if ( checkCharge ) charge = abs(q);
  assert( dauCharge_.size() == 3 );
}
	 
bool ThreeBodyCombiner::preselect( const Candidate & c1, const Candidate & c2,
				   const Candidate & c3 ) const {
  if ( checkCharge ) {
    if ( charge != abs(c1.charge() + c2.charge() + c3.charge()) ) return false;
    if ( abs(dauCharge_[0]) != abs(c1.charge()) ) return false;
    if ( abs(dauCharge_[1]) != abs(c2.charge()) ) return false;
    if ( abs(dauCharge_[2]) != abs(c3.charge()) ) return false;
  }

  if ( overlap( c1, c2 ) || overlap( c1, c3 ) || overlap( c2, c3 ) ) return false;

  return true;
}

Candidate * ThreeBodyCombiner::combine( const Candidate & c1, const Candidate & c2,
					const Candidate & c3 ) {
  CompositeCandidate * cmp( new CompositeCandidate );
  cmp->addDaughter( c1 );
  cmp->addDaughter( c2 );
  cmp->addDaughter( c3 );
  cmp->set( addp4 );
  return cmp;
}

auto_ptr<CandidateCollection> 
ThreeBodyCombiner::combine( const CandidateCollection * src1, const CandidateCollection * src2,
			    const CandidateCollection * src3 ) {

  auto_ptr<CandidateCollection> comps( new CandidateCollection );

  if ( src1 == src2 && src1 == src3 ) {

    cout << "CASE 1: all equal coll" << endl;
    const CandidateCollection & cands = * src1;
    const int n = cands.size();
    for( int i1 = 0; i1 < n; ++ i1 ) {
      const Candidate & c1 = cands[ i1 ];
      for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
	const Candidate & c2 = cands[ i2 ];
	for( int i3 = i2 + 1; i3 < n; ++ i3 ) {
	  const Candidate & c3 = cands[ i3 ];
	  if ( preselect( c1, c2, c3 ) ) {
	    std::auto_ptr<Candidate> c( combine( c1, c2, c3 ) );
	    if ( (*select)( * c ) )
	      comps->push_back( c.release() );
	  }
	}
      }
    }
  } else if ( src1 == src2 ) {
    cout << "CASE 2: coll1 = coll2" << endl;
    combineWithTwoEqualCollection( src1, src3, comps );
  } else if ( src1 == src3 ) {
    cout << "CASE 3: coll1 = coll3" << endl;
    combineWithTwoEqualCollection( src1, src2, comps );
  } else if ( src2 == src3 ) {
    cout << "CASE 4: coll2 = coll3" << endl;
    combineWithTwoEqualCollection( src2, src1, comps );
  } else {
    cout << "CASE 5: all different coll" << endl;
    const CandidateCollection & cands1 = * src1, & cands2 = * src2,
      & cands3 = * src3;
    const int n1 = cands1.size(), n2 = cands2.size(), n3 = cands3.size();
    for( int i1 = 0; i1 < n1; ++ i1 ) {
      const Candidate & c1 = cands1[ i1 ];
      for ( int i2 = 0; i2 < n2; ++ i2 ) {
	const Candidate & c2 = cands2[ i2 ];
	for ( int i3 = 0; i3 < n3; ++ i3 ) {
	  const Candidate & c3 = cands3[ i3 ];
	  if ( preselect( c1, c2, c3 ) ) {
	    std::auto_ptr<Candidate> c( combine( c1, c2, c3 ) );
	    if ( (*select)( * c ) )
	      comps->push_back( c.release() );
	  }
	}
      }
    }
  }

  return comps;

}


void ThreeBodyCombiner::combineWithTwoEqualCollection( const CandidateCollection * equalSrc,
						       const CandidateCollection * diffSrc,
						       auto_ptr<CandidateCollection> comps ) {
  const CandidateCollection & cands = * equalSrc;
  const CandidateCollection & candsDiff = * diffSrc;
  const int n = cands.size();
  const int n3 = candsDiff.size();
  for( int i1 = 0; i1 < n; ++ i1 ) {
    const Candidate & c1 = cands[ i1 ];
    for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
      const Candidate & c2 = cands[ i2 ];
      for( int i3 = 0; i3 < n3; ++ i3 ) {
	const Candidate & c3 = candsDiff[ i3 ];
	if ( preselect( c1, c2, c3 ) ) {
	  std::auto_ptr<Candidate> c( combine( c1, c2, c3 ) );
	  if ( (*select)( * c ) )
	    comps->push_back( c.release() );
	}
      }
    }
  }
}
