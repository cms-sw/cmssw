// $Id: reftobasevector_t.cppunit.cc,v 1.2 2007/04/10 11:46:17 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>

class testRefToBaseVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testRefToBaseVector);
  CPPUNIT_TEST(check);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void check();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testRefToBaseVector);
namespace testreftobase {
  struct Base {
    virtual ~Base() {}
    virtual int val() const=0;
  };  

  struct Inherit1 : public Base {
    virtual int val() const { return 1;}
  };
  struct Inherit2 : public Base {
    virtual int val() const {return 2;}
  };
  
  template<class T>
    struct TestHandle {
      TestHandle(const edm::ProductID& iId, const T* iProd) : id_(iId), prod_(iProd) {}
      const edm::ProductID& id() const { return id_;}
      const T* product() const { return prod_;}
    private:
      edm::ProductID id_;
      const T* prod_;
    };
}

using namespace testreftobase;

void
testRefToBaseVector::check()
{
  using namespace edm;

  std::vector<Inherit1> v1(2,Inherit1());
  std::vector<Inherit2> v2(2,Inherit2());
  
  TestHandle<std::vector<Inherit1> > h1(ProductID(1), &v1);
  RefVector<std::vector<Inherit1> > rv1;
  rv1.push_back( Ref<std::vector<Inherit1> >( h1, 0 ) );
  rv1.push_back( Ref<std::vector<Inherit1> >( h1, 1 ) );
  TestHandle<std::vector<Inherit2> > h2(ProductID(2), &v2);
  RefVector<std::vector<Inherit2> > rv2;
  rv2.push_back( Ref<std::vector<Inherit2> >( h2, 0 ) );
  rv2.push_back( Ref<std::vector<Inherit2> >( h2, 1 ) );

  RefToBaseVector<Base> bv1( rv1 );
  RefToBase<Base> r1_0 = bv1[ 0 ];
  RefToBase<Base> r1_1 = bv1[ 1 ];
  RefToBaseVector<Base> bv2( rv2 );
  RefToBase<Base> r2_0 = bv2[ 0 ];
  RefToBase<Base> r2_1 = bv2[ 1 ];

  CPPUNIT_ASSERT( bv1.size() == 2 );
  CPPUNIT_ASSERT( bv2.size() == 2 );
  CPPUNIT_ASSERT( r1_0->val() == 1 );
  CPPUNIT_ASSERT( r1_1->val() == 1 );
  CPPUNIT_ASSERT( r2_0->val() == 2 );
  CPPUNIT_ASSERT( r2_1->val() == 2 );

  RefToBaseVector<Base>::const_iterator b = bv1.begin(), e = bv1.end();
  RefToBaseVector<Base>::const_iterator i = b; 
  CPPUNIT_ASSERT( i->val() == 1 );
  CPPUNIT_ASSERT( i != e );
  CPPUNIT_ASSERT( i - b == 0 );
  ++i;
  CPPUNIT_ASSERT( i->val() == 1 );
  CPPUNIT_ASSERT( i != e );
  CPPUNIT_ASSERT( i - b == 1 );
  ++ i;
  CPPUNIT_ASSERT( i == e );

  RefToBaseVector<Base> assigned_from_bv1;
  assigned_from_bv1 = bv1;
  CPPUNIT_ASSERT(assigned_from_bv1.size() == bv1.size());
  //CPPUNIT_ASSERT(assigned_from_bv1 == bv1);
}
