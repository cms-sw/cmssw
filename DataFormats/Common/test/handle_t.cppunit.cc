#include "cppunit/extensions/HelperMacros.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"
#include "FWCore/Utilities/interface/Exception.h"

class testHandle : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testHandle);
  CPPUNIT_TEST(check);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void check();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testHandle);

void
testHandle::check()
{
  using namespace edm;
  
  {
    Handle<int> hDefault;
    CPPUNIT_ASSERT(not hDefault.isValid());
    CPPUNIT_ASSERT(not hDefault.failedToGet());
    
    //the following leads to a seg fault :(
    //CPPUNIT_ASSERT(hDefault.id() == ProductID{});
    CPPUNIT_ASSERT(not hDefault.whyFailed());
    
    //This doesn't throw
    //CPPUNIT_ASSERT_THROW([&hDefault](){*hDefault;},
    //                     cms::Exception);
  }

  {
    Provenance provDummy;
    int value = 3;
    
    Handle<int> h(&value,&provDummy);
    CPPUNIT_ASSERT(h.isValid());
    CPPUNIT_ASSERT(not h.failedToGet());
    CPPUNIT_ASSERT(not h.whyFailed());
    CPPUNIT_ASSERT(3 == *h);
    
    Handle<int> hCopy(h);
    CPPUNIT_ASSERT(hCopy.isValid());
    CPPUNIT_ASSERT(not hCopy.failedToGet());
    CPPUNIT_ASSERT(not hCopy.whyFailed());
    CPPUNIT_ASSERT(3 == *hCopy);
    
    Handle<int> hOpEq;
    hOpEq = h;
    CPPUNIT_ASSERT(hOpEq.isValid());
    CPPUNIT_ASSERT(not hOpEq.failedToGet());
    CPPUNIT_ASSERT(not hOpEq.whyFailed());
    CPPUNIT_ASSERT(3 == *hOpEq);

    Handle<int> hOpEqMove;
    hOpEqMove = std::move(hCopy);
    CPPUNIT_ASSERT(hOpEqMove.isValid());
    CPPUNIT_ASSERT(not hOpEqMove.failedToGet());
    CPPUNIT_ASSERT(not hOpEqMove.whyFailed());
    CPPUNIT_ASSERT(3 == *hOpEqMove);

  }
  
  {
    Handle<int> hFail(makeHandleExceptionFactory([]()->std::shared_ptr<cms::Exception> {
      return std::make_shared<cms::Exception>("DUMMY");
    }));
    
    CPPUNIT_ASSERT(not hFail.isValid());
    CPPUNIT_ASSERT(hFail.failedToGet());

    CPPUNIT_ASSERT(hFail.whyFailed());

    CPPUNIT_ASSERT_THROW(*hFail,
                         cms::Exception);

  }
  
}