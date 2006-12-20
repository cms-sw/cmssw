#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/Framework/interface/View.h"

class testView: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testView);
  CPPUNIT_TEST(basic);
  CPPUNIT_TEST(createFromArray);
  CPPUNIT_TEST(directAccess);
  CPPUNIT_TEST(iterateForward);
  CPPUNIT_TEST(iterateBackward);
  CPPUNIT_TEST_SUITE_END();

 public:
  testView() {}
  ~testView() {}
  void setUp() {}
  void tearDown() {}

  void basic();
  void createFromArray();
  void directAccess();
  void iterateForward();
  void iterateBackward();

 private:
  typedef int  value_type;
  typedef edm::View<value_type> view;
  typedef view::const_iterator const_iterator;
  typedef view::const_reverse_iterator const_reverse_iterator;
  typedef view::const_reference const_reference;
  typedef view::size_type size_type;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testView);

void testView::basic()
{
  view v1;
  CPPUNIT_ASSERT(v1.size() == 0);
  CPPUNIT_ASSERT(v1.empty());
  view v2(v1);
  CPPUNIT_ASSERT(v1==v2);
  view v3;
  v3 = v1;
  CPPUNIT_ASSERT(v3==v1);

  CPPUNIT_ASSERT(! (v1 < v2));
  CPPUNIT_ASSERT(v1 <= v2);
  CPPUNIT_ASSERT(! (v1 > v2));
  CPPUNIT_ASSERT(v1 >= v2);
  CPPUNIT_ASSERT(! (v1 != v2));
}

void testView::createFromArray()
{
  value_type vals[] = { 1, 2, 3, 4, 5 };
  size_t sz = sizeof(vals)/sizeof(value_type);

  view v1;
  edm::fill_from_range(vals, vals+sz, v1);
  CPPUNIT_ASSERT(v1.size() == 5);
  view v2;
  edm::fill_from_range(vals, vals+sz, v2);
  CPPUNIT_ASSERT(v1==v2);
}

void testView::directAccess()
{
  value_type vals[] = { 1, 2, 3, 4, 5 };
  size_t sz = sizeof(vals)/sizeof(value_type);

  view v1;
  edm::fill_from_range(vals, vals+sz, v1);
  for (size_type i = 0; i < v1.size(); ++i)
    {
      CPPUNIT_ASSERT(v1[i] == vals[i]);
    }  
}

void testView::iterateForward()
{
  value_type vals[] = { 1, 2, 3, 4, 5 };
  size_t sz = sizeof(vals)/sizeof(value_type);

  view v1;
  edm::fill_from_range(vals, vals+sz, v1);

  const_iterator i = v1.begin();
  CPPUNIT_ASSERT( *i == 1 );
  ++i;
  CPPUNIT_ASSERT( *i == 2 );
}

void testView::iterateBackward()
{
  value_type vals[] = { 1, 2, 3, 4, 5 };
  size_t sz = sizeof(vals)/sizeof(value_type);

  view v1;
  edm::fill_from_range(vals, vals+sz, v1);

  const_reverse_iterator i = v1.rbegin();
  CPPUNIT_ASSERT( *i == 5 );
  ++i;
  CPPUNIT_ASSERT( *i == 4 );  
}
