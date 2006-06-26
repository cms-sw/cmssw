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
  reco::TrackBase::CovarianceMatrix cov( e, e + sizeof( e ) / sizeof( double ) );
  reco::Track t( chi2, ndof, par, 3.14, cov );
  CPPUNIT_ASSERT( t.chi2() == chi2 );
  CPPUNIT_ASSERT( t.ndof() == ndof );
  CPPUNIT_ASSERT( t.normalizedChi2() == chi2 / ndof );
  CPPUNIT_ASSERT( t.parameters() == par );
  CPPUNIT_ASSERT( t.covariance() == cov );
}

