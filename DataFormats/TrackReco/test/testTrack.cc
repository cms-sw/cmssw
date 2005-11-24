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
  const int ndof = 10, found = 23, invalid = 45, lost = 67;
  // assume helix already tested
  reco::Track::Parameters h;
  reco::Track::Covariance c;
  reco::Track t( chi2, ndof, found, lost, invalid, h, c );
  CPPUNIT_ASSERT( t.chi2() == chi2 );
  CPPUNIT_ASSERT( t.ndof() == ndof );
  CPPUNIT_ASSERT( t.normalizedChi2() == chi2 / ndof );
  CPPUNIT_ASSERT( t.foundHits() == found );
  CPPUNIT_ASSERT( t.lostHits() == lost );
  CPPUNIT_ASSERT( t.invalidHits() == invalid );
}

