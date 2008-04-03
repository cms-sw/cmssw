#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Simplify.h"
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

class testSimplifications : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSimplifications);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSimplifications);

void testSimplifications::checkAll() {
  using namespace funct;
  using namespace boost;
  BOOST_STATIC_ASSERT((is_same<
		       Sum<Numerical<2>, Numerical<3> >::type, 
		       Numerical<5>
		       >::value));
  
  BOOST_STATIC_ASSERT((is_same<
		       Difference<Numerical<2>, Numerical<3> >::type, 
		       Numerical<-1>
		       >::value));
  BOOST_STATIC_ASSERT((is_same<
		       Product<Numerical<2>, Numerical<3> >::type, 
		       Numerical<6>
		       >::value));
  BOOST_STATIC_ASSERT((is_same<
		       Minus<Numerical<6> >::type, 
		       Numerical<-6>
		       >::value));

}
