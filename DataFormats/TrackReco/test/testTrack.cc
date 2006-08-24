#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/TrackReco/interface/Track.h"

class testTrack : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTrack);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testTrack);

void testTrack::checkAll() {
  const double chi2 = 20.0;
  const int ndof = 10;
  double p[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  double e[] = { 1.1, 
		 1.2, 2.2, 
		 1.3, 2.3, 3.3,
		 1.4, 2.4, 3.4, 4.4,
		 1.5, 2.5, 3.5, 4.5, 5.5 };
  reco::TrackBase::ParameterVector par( p[0], p[1], p[2], p[3], p[4] );
  reco::TrackBase::CovarianceMatrix cov( e, e + 15 );
  reco::Track t( chi2, ndof, par, 3.14, cov );
  CPPUNIT_ASSERT( t.chi2() == chi2 );
  CPPUNIT_ASSERT( t.ndof() == ndof );
  CPPUNIT_ASSERT( t.normalizedChi2() == chi2 / ndof );
  CPPUNIT_ASSERT( t.parameters() == par );
  CPPUNIT_ASSERT( t.covariance() == cov );

  int k = 0;
  CPPUNIT_ASSERT( cov( 0, 0 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 1, 0 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 1, 1 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 2, 0 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 2, 1 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 2, 2 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 3, 0 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 3, 1 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 3, 2 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 3, 3 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 4, 0 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 4, 1 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 4, 2 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 4, 3 ) == e[ k ++ ] );
  CPPUNIT_ASSERT( cov( 4, 4 ) == e[ k ++ ] );

  CPPUNIT_ASSERT( t.covariance( 0, 0 ) == cov( 0, 0 ) );
  CPPUNIT_ASSERT( t.covariance( 0, 1 ) == cov( 0, 1 ) );
  CPPUNIT_ASSERT( t.covariance( 0, 2 ) == cov( 0, 2 ) );
  CPPUNIT_ASSERT( t.covariance( 0, 3 ) == cov( 0, 3 ) );
  CPPUNIT_ASSERT( t.covariance( 0, 4 ) == cov( 0, 4 ) );
  CPPUNIT_ASSERT( t.covariance( 1, 1 ) == cov( 1, 1 ) );
  CPPUNIT_ASSERT( t.covariance( 1, 2 ) == cov( 1, 2 ) );
  CPPUNIT_ASSERT( t.covariance( 1, 3 ) == cov( 1, 3 ) );
  CPPUNIT_ASSERT( t.covariance( 1, 4 ) == cov( 1, 4 ) );
  CPPUNIT_ASSERT( t.covariance( 2, 2 ) == cov( 2, 2 ) );
  CPPUNIT_ASSERT( t.covariance( 2, 3 ) == cov( 2, 3 ) );
  CPPUNIT_ASSERT( t.covariance( 2, 4 ) == cov( 2, 4 ) );
  CPPUNIT_ASSERT( t.covariance( 3, 3 ) == cov( 3, 3 ) );
  CPPUNIT_ASSERT( t.covariance( 3, 4 ) == cov( 3, 4 ) );
  CPPUNIT_ASSERT( t.covariance( 4, 4 ) == cov( 4, 4 ) );
}

