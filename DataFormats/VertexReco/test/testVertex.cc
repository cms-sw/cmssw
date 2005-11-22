#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/VertexReco/interface/Vertex.h"

class testVertex : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testVertex);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testVertex);

void testVertex::checkAll() {
  const double chi2 = 20.0;
  const int ndof = 10;
  const size_t size = 0; // can't test refs at the moment
  const double x = 1.0, y = 2.0, z = 3.0;
  const double ee[ reco::Vertex::Error::size ] = { 1.1, 1.2, 1.3, 2.2, 2.3, 3.3 };
  reco::Vertex::Error err( ee );
  reco::Vertex t( chi2, ndof, x, y, z, err, size );
  CPPUNIT_ASSERT( t.tracksSize() == 0 );
  CPPUNIT_ASSERT( t.chi2() == chi2 );
  CPPUNIT_ASSERT( t.ndof() == ndof );
  CPPUNIT_ASSERT( t.normalizedChi2() == chi2 / ndof );
  CPPUNIT_ASSERT( t.x() == x );
  CPPUNIT_ASSERT( t.y() == y );
  CPPUNIT_ASSERT( t.z() == z );
  const reco::Vertex::Error & e = t.error();
  const double 
    e00 = e.get<0,0>(), e01 = e.get<0,1>(), e02 = e.get<0,2>(),
    e10 = e.get<1,0>(), e11 = e.get<1,1>(), e12 = e.get<1,2>(),
    e20 = e.get<2,0>(), e21 = e.get<2,1>(), e22 = e.get<2,2>();
  CPPUNIT_ASSERT( e00 == ee[ 0 ] );
  CPPUNIT_ASSERT( e01 == ee[ 1 ] );
  CPPUNIT_ASSERT( e02 == ee[ 2 ] );

  CPPUNIT_ASSERT( e10 == ee[ 1 ] );
  CPPUNIT_ASSERT( e11 == ee[ 3 ] );
  CPPUNIT_ASSERT( e12 == ee[ 4 ] );

  CPPUNIT_ASSERT( e20 == ee[ 2 ] );
  CPPUNIT_ASSERT( e21 == ee[ 4 ] );
  CPPUNIT_ASSERT( e22 == ee[ 5 ] );
}
