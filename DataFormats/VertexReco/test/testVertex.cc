#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/VertexReco/interface/Vertex.h"

class testVertex : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testVertex);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testVertex);

void testVertex::checkAll() {
  const double chi2 = 20.0;
  const double ndof = 10;
  const size_t size = 0;  // can't test refs at the moment
  const double x = 1.0, y = 2.0, z = 3.0;
  const float ee[reco::Vertex::Error::kSize] = {1.1, 1.2, 1.3, 2.2, 2.3, 3.3};
  reco::Vertex::Error err;
  int k = 0;
  for (int i = 0; i < reco::Vertex::Error::kRows; ++i)
    for (int j = i; j < reco::Vertex::Error::kCols; ++j)
      err(i, j) = ee[k++];

  reco::Vertex t(reco::Vertex::Point(x, y, z), err, chi2, ndof, size);
  CPPUNIT_ASSERT(t.tracksSize() == 0);
  CPPUNIT_ASSERT(t.chi2() == chi2);
  CPPUNIT_ASSERT(t.ndof() == ndof);
  CPPUNIT_ASSERT(t.normalizedChi2() == chi2 / ndof);
  CPPUNIT_ASSERT(t.x() == x);
  CPPUNIT_ASSERT(t.y() == y);
  CPPUNIT_ASSERT(t.z() == z);
  const reco::Vertex::Error e = t.covariance();
  const float e00 = e(0, 0), e01 = e(0, 1), e02 = e(0, 2), e10 = e(1, 0), e11 = e(1, 1), e12 = e(1, 2), e20 = e(2, 0),
              e21 = e(2, 1), e22 = e(2, 2);

  CPPUNIT_ASSERT(e00 == ee[0]);
  CPPUNIT_ASSERT(e01 == ee[1]);
  CPPUNIT_ASSERT(e02 == ee[2]);

  CPPUNIT_ASSERT(e10 == ee[1]);
  CPPUNIT_ASSERT(e11 == ee[3]);
  CPPUNIT_ASSERT(e12 == ee[4]);

  CPPUNIT_ASSERT(e20 == ee[2]);
  CPPUNIT_ASSERT(e21 == ee[4]);
  CPPUNIT_ASSERT(e22 == ee[5]);
}
