// $Id: testAssociationVector.cc,v 1.8 2007/06/14 20:53:12 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/test/TestHandle.h"
using namespace edm;

class testAssociationNew : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testAssociationNew);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testAssociationNew);

void testAssociationNew::checkAll() {
  using namespace std;
  using namespace edm;

  typedef vector<double> CVal;
  typedef vector<int> CKey1;
  typedef vector<float> CKey2;

  CVal k;
  k.push_back(1.1);
  k.push_back(2.2);
  k.push_back(3.3);
  ProductID const pidV(1);
  TestHandle<CVal> handleV(&k, pidV);

  CKey1 v1;
  v1.push_back(1);
  v1.push_back(2);
  v1.push_back(3);
  v1.push_back(4);
  ProductID const pidK1(2);
  TestHandle<CKey1> handleK1(&v1, pidK1);

  CKey2 v2;
  v2.push_back(10.);
  v2.push_back(20.);
  v2.push_back(30.);
  v2.push_back(40.);
  v2.push_back(50.);
  ProductID const pidK2(3);
  TestHandle<CKey2> handleK2(&v2, pidK2);
  const int w1[4] = { 2, 1, 0, 2 };
  const int w2[5] = { 1, 0, 2, 1, -1 };

  {
  edm::Association<CVal> assoc(handleV);
  edm::Association<CVal>::Filler filler(assoc);
  filler.insert(handleK1, w1, w1 + 4);
  filler.insert(handleK2, w2, w2 + 5);
  filler.fill();
  edm::Ref<CVal> r1 = assoc[edm::Ref<CKey1>(handleK1, 0)];
  edm::Ref<CVal> r2 = assoc[edm::Ref<CKey1>(handleK1, 1)];
  edm::Ref<CVal> r3 = assoc[edm::Ref<CKey1>(handleK1, 2)];
  edm::Ref<CVal> r4 = assoc[edm::Ref<CKey1>(handleK1, 3)];
  CPPUNIT_ASSERT(r1.isNonnull());
  CPPUNIT_ASSERT(r2.isNonnull());
  CPPUNIT_ASSERT(r3.isNonnull());
  CPPUNIT_ASSERT(r4.isNonnull());
  CPPUNIT_ASSERT( *r1 == k[w1[0]] );
  CPPUNIT_ASSERT( *r2 == k[w1[1]] );
  CPPUNIT_ASSERT( *r3 == k[w1[2]] );
  CPPUNIT_ASSERT( *r4 == k[w1[3]] );
  edm::Ref<CVal> s1 = assoc[edm::Ref<CKey2>(handleK2, 0)];
  edm::Ref<CVal> s2 = assoc[edm::Ref<CKey2>(handleK2, 1)];
  edm::Ref<CVal> s3 = assoc[edm::Ref<CKey2>(handleK2, 2)];
  edm::Ref<CVal> s4 = assoc[edm::Ref<CKey2>(handleK2, 3)];
  edm::Ref<CVal> s5 = assoc[edm::Ref<CKey2>(handleK2, 4)];
  CPPUNIT_ASSERT(s1.isNonnull());
  CPPUNIT_ASSERT(s2.isNonnull());
  CPPUNIT_ASSERT(s3.isNonnull());
  CPPUNIT_ASSERT(s4.isNonnull());
  CPPUNIT_ASSERT(s5.isNull());
  CPPUNIT_ASSERT( *s1 == k[w2[0]] );
  CPPUNIT_ASSERT( *s2 == k[w2[1]] );
  CPPUNIT_ASSERT( *s3 == k[w2[2]] );
  CPPUNIT_ASSERT( *s4 == k[w2[3]] );
  CPPUNIT_ASSERT( assoc.size() == 9 );
  }

  {
  edm::Association<CVal> assoc(handleV);
  edm::Association<CVal>::Filler filler1(assoc);
  filler1.insert(handleK1, w1, w1 + 4);
  filler1.fill();
  edm::Association<CVal>::Filler filler2(assoc);
  filler2.insert(handleK2, w2, w2 + 5);
  filler2.fill();
  edm::Ref<CVal> r1 = assoc[edm::Ref<CKey1>(handleK1, 0)];
  edm::Ref<CVal> r2 = assoc[edm::Ref<CKey1>(handleK1, 1)];
  edm::Ref<CVal> r3 = assoc[edm::Ref<CKey1>(handleK1, 2)];
  edm::Ref<CVal> r4 = assoc[edm::Ref<CKey1>(handleK1, 3)];
  CPPUNIT_ASSERT(r1.isNonnull());
  CPPUNIT_ASSERT(r2.isNonnull());
  CPPUNIT_ASSERT(r3.isNonnull());
  CPPUNIT_ASSERT(r4.isNonnull());
  CPPUNIT_ASSERT( *r1 == k[w1[0]] );
  CPPUNIT_ASSERT( *r2 == k[w1[1]] );
  CPPUNIT_ASSERT( *r3 == k[w1[2]] );
  CPPUNIT_ASSERT( *r4 == k[w1[3]] );
  edm::Ref<CVal> s1 = assoc[edm::Ref<CKey2>(handleK2, 0)];
  edm::Ref<CVal> s2 = assoc[edm::Ref<CKey2>(handleK2, 1)];
  edm::Ref<CVal> s3 = assoc[edm::Ref<CKey2>(handleK2, 2)];
  edm::Ref<CVal> s4 = assoc[edm::Ref<CKey2>(handleK2, 3)];
  edm::Ref<CVal> s5 = assoc[edm::Ref<CKey2>(handleK2, 4)];
  CPPUNIT_ASSERT(s1.isNonnull());
  CPPUNIT_ASSERT(s2.isNonnull());
  CPPUNIT_ASSERT(s3.isNonnull());
  CPPUNIT_ASSERT(s4.isNonnull());
  CPPUNIT_ASSERT(s5.isNull());
  CPPUNIT_ASSERT( *s1 == k[w2[0]] );
  CPPUNIT_ASSERT( *s2 == k[w2[1]] );
  CPPUNIT_ASSERT( *s3 == k[w2[2]] );
  CPPUNIT_ASSERT( *s4 == k[w2[3]] );
  CPPUNIT_ASSERT( assoc.size() == 9 );
  }
  {
  edm::Association<CVal> assoc1(handleV);
  edm::Association<CVal>::Filler filler1(assoc1);
  filler1.insert(handleK1, w1, w1 + 4);
  filler1.fill();
  edm::Association<CVal> assoc2(handleV);
  edm::Association<CVal>::Filler filler2(assoc2);
  filler2.insert(handleK2, w2, w2 + 5);
  filler2.fill();
  edm::Association<CVal> assoc = assoc1 + assoc2;

  edm::Ref<CVal> r1 = assoc[edm::Ref<CKey1>(handleK1, 0)];
  edm::Ref<CVal> r2 = assoc[edm::Ref<CKey1>(handleK1, 1)];
  edm::Ref<CVal> r3 = assoc[edm::Ref<CKey1>(handleK1, 2)];
  edm::Ref<CVal> r4 = assoc[edm::Ref<CKey1>(handleK1, 3)];
  CPPUNIT_ASSERT(r1.isNonnull());
  CPPUNIT_ASSERT(r2.isNonnull());
  CPPUNIT_ASSERT(r3.isNonnull());
  CPPUNIT_ASSERT(r4.isNonnull());
  CPPUNIT_ASSERT( *r1 == k[w1[0]] );
  CPPUNIT_ASSERT( *r2 == k[w1[1]] );
  CPPUNIT_ASSERT( *r3 == k[w1[2]] );
  CPPUNIT_ASSERT( *r4 == k[w1[3]] );
  edm::Ref<CVal> s1 = assoc[edm::Ref<CKey2>(handleK2, 0)];
  edm::Ref<CVal> s2 = assoc[edm::Ref<CKey2>(handleK2, 1)];
  edm::Ref<CVal> s3 = assoc[edm::Ref<CKey2>(handleK2, 2)];
  edm::Ref<CVal> s4 = assoc[edm::Ref<CKey2>(handleK2, 3)];
  edm::Ref<CVal> s5 = assoc[edm::Ref<CKey2>(handleK2, 4)];
  CPPUNIT_ASSERT(s1.isNonnull());
  CPPUNIT_ASSERT(s2.isNonnull());
  CPPUNIT_ASSERT(s3.isNonnull());
  CPPUNIT_ASSERT(s4.isNonnull());
  CPPUNIT_ASSERT(s5.isNull());
  CPPUNIT_ASSERT( *s1 == k[w2[0]] );
  CPPUNIT_ASSERT( *s2 == k[w2[1]] );
  CPPUNIT_ASSERT( *s3 == k[w2[2]] );
  CPPUNIT_ASSERT( *s4 == k[w2[3]] );
  CPPUNIT_ASSERT( assoc.size() == 9 );
  }
}

