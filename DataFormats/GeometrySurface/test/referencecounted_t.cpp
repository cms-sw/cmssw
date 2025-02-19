#include <cassert>

#include <cppunit/extensions/HelperMacros.h>

//#include "boost/intrusive_ptr.hpp"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

namespace testreferencecounted {

   class Test: public CppUnit::TestFixture
   {
	 CPPUNIT_TEST_SUITE(Test);
	 CPPUNIT_TEST(deleteTest);
	 CPPUNIT_TEST(multiRefTest);
	 CPPUNIT_TEST(assignmentTest);
	 CPPUNIT_TEST(intrusiveTest);
	 CPPUNIT_TEST_SUITE_END();
      public:
	 void setUp(){}
	 void tearDown(){}
	 void deleteTest();
	 void multiRefTest();
	 void assignmentTest();
	 void intrusiveTest();
   };
   
   static unsigned int s_construct = 0;

   struct RefTest : public ReferenceCountedPoolAllocated {
	 RefTest() { s_construct = 1; }
	 ~RefTest() { s_construct = 0; }
   };

   typedef ReferenceCountingPointer<RefTest> RefPtr;
}
///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testreferencecounted::Test);

namespace testreferencecounted {
   void Test::deleteTest() {
      {
	 RefPtr pointer( new RefTest );
      }
      assert( 0 == s_construct );
   }

   void Test::multiRefTest() {
      {
	 RefPtr pointer( new RefTest );
	 {
	    RefPtr pointer2( pointer );
	 }
      }
      assert( 0 == s_construct );
   }

   void Test::assignmentTest() {
      {
	 RefPtr pointer( new RefTest );
	 {
	    RefPtr pointer2;
	    pointer2 = pointer;
	 }
      }
      assert( 0 == s_construct );
   }

   void Test::intrusiveTest() {
      {
	 RefTest* ptr = new RefTest;
	 RefPtr pointer( ptr );
	 {
	    RefPtr pointer2(ptr );
	 }
      }
      assert( 0 == s_construct );
   }
}


///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testreferencecounted::Test);

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
