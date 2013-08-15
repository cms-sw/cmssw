#include "cppunit/extensions/HelperMacros.h"
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/TestHandle.h"
using namespace edm;

class testAssociationVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testAssociationVector);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testAssociationVector);

void testAssociationVector::checkAll() {
  typedef std::vector<double> CKey;
  typedef std::vector<int> CVal;

  CKey k;
  k.push_back(1.1);
  k.push_back(2.2);
  k.push_back(3.3);
  ProductID const pid(1, 1);
  TestHandle<CKey> handle(&k, pid);
  RefProd<CKey> ref(handle);
  AssociationVector<RefProd<CKey>, CVal> v(ref);
  v.setValue(0, 1);
  v.setValue(1, 2);
  v.setValue(2, 3);
  CPPUNIT_ASSERT(v.size() == 3);
  CPPUNIT_ASSERT(v.keyProduct() == ref);
  CPPUNIT_ASSERT(v[0].second == 1);
  CPPUNIT_ASSERT(v[1].second == 2);
  CPPUNIT_ASSERT(v[2].second == 3);
  CPPUNIT_ASSERT(*v[0].first == 1.1);
  CPPUNIT_ASSERT(*v[1].first == 2.2);
  CPPUNIT_ASSERT(*v[2].first == 3.3);
  CPPUNIT_ASSERT(*v.key(0) == 1.1);
  CPPUNIT_ASSERT(*v.key(1) == 2.2);
  CPPUNIT_ASSERT(*v.key(2) == 3.3);
  Ref<CKey> rc0( handle, 0 ), rc1( handle, 1 ), rc2( handle, 2 );
  CPPUNIT_ASSERT( v[rc0] == 1 );
  CPPUNIT_ASSERT( v[rc1] == 2 );
  CPPUNIT_ASSERT( v[rc2] == 3 );
  ProductID const assocPid(1, 2);
  TestHandle<AssociationVector<RefProd<CKey>, CVal> > assocHandle(&v, assocPid); 
  Ref<AssociationVector<RefProd<CKey>, CVal> > r1( assocHandle, 0 );
  CPPUNIT_ASSERT(*r1->first == 1.1);
  CPPUNIT_ASSERT(r1->second == 1);
  Ref<AssociationVector<RefProd<CKey>, CVal> > r2( assocHandle, 1 );
  CPPUNIT_ASSERT(*r2->first == 2.2);
  CPPUNIT_ASSERT(r2->second == 2);
  Ref<AssociationVector<RefProd<CKey>, CVal> > r3( assocHandle, 2 );
  CPPUNIT_ASSERT(*r3->first == 3.3);
  CPPUNIT_ASSERT(r3->second == 3);
  v[rc0] = 111;
  v[rc1] = 122;
  v[rc2] = 133;
  CPPUNIT_ASSERT( v[rc0] == 111 );
  CPPUNIT_ASSERT( v[rc1] == 122 );
  CPPUNIT_ASSERT( v[rc2] == 133 );
}

