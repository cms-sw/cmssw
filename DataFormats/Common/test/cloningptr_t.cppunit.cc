// $Id: cloningptr_t.cppunit.cc,v 1.2 2007/08/06 22:16:53 wmtan Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Common/interface/CloningPtr.h"

#include <vector>

class testCloningPtr : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCloningPtr);
  CPPUNIT_TEST(check);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void check();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testCloningPtr);
namespace testcloningptr {
  struct Base
  {
    virtual ~Base() {}
    virtual int val() const=0;
    virtual Base* clone() const = 0;
  };  

  struct Inherit : public Base {
    Inherit(int iValue):val_(iValue) {}
    virtual int val() const { return val_;}
    virtual Base* clone() const { return new Inherit(*this);}
    int val_;
  };  
}

using namespace testcloningptr;

void
testCloningPtr::check()
{
  using namespace edm;

  Inherit one(1);
  CloningPtr<Base> cpOne(one);
  CPPUNIT_ASSERT(&one != cpOne.get());
  CPPUNIT_ASSERT(1==cpOne->val());
  CPPUNIT_ASSERT(1==(*cpOne).val());
  
  
  CloningPtr<Base> cpOtherOne(cpOne);
  CPPUNIT_ASSERT(cpOne.get() != cpOtherOne.get());
  CPPUNIT_ASSERT(cpOtherOne->val()==1);
  
  CloningPtr<Base> eqOne;
  eqOne=cpOne;

  CPPUNIT_ASSERT(cpOne.get() != eqOne.get());
  CPPUNIT_ASSERT(eqOne->val()==1);}
