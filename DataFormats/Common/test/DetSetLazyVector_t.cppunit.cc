// $Id: DetSetLazyVector_t.cppunit.cc,v 1.4 2007/01/23 00:25:53 wmtan Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Common/interface/DetSetLazyVector.h"
#include "DataFormats/Common/interface/DetSetVector.h"


class testDetSetLazyVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDetSetLazyVector);
  CPPUNIT_TEST(checkConstruction);
  CPPUNIT_TEST(checkFind);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkConstruction();
  void checkFind();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDetSetLazyVector);
namespace testdetsetlazyvector {
  class Value
  {
public:
    // VALUES must be default constructible
    Value() : d_(0.0) { }

    // This constructor is used for testing; it is not required by the
    // concept VALUE.
    explicit Value(double d) : d_(d) { }

    // This access function is used for testing; it is not required by
    // the concept VALUE.
    double val() const { return d_; }

    // VALUES must be destructible
    ~Value() {}


    // VALUES must be LessThanComparable
    bool operator< (Value const& other) const
  { return d_ < other.d_; }

    // The private stuff below is all implementation detail, and not
    // required by the concept VALUE.
private:
    double        d_;
  };

  struct TestGetter : public edm::dslv::LazyGetter<Value> {

    TestGetter(const edm::DetSetVector<Value>& iValue) : values_(iValue) {}
    void fill(edm::DetSet<Value>& oSet) {
      oSet.data = values_.find(oSet.id)->data;
    }
    const edm::DetSetVector< Value >& values_;
  };
}

using namespace testdetsetlazyvector;
typedef edm::DetSetVector<Value> dsv_type;
typedef dsv_type::detset        detset;

void
testDetSetLazyVector::checkConstruction()
{
  dsv_type c;
  detset    d3;
  Value v1(1.1);
  Value v2(2.2);
  d3.id = edm::det_id_type(3);
  d3.data.push_back(v1);
  d3.data.push_back(v2);
  c.insert(d3);
  detset    d1;
  Value v1a(4.1);
  Value v2a(3.2);
  d1.id = edm::det_id_type(1);
  d1.data.push_back(v1a);
  d1.data.push_back(v2a);
  c.insert(d1);
  c.post_insert();

  boost::shared_ptr<edm::dslv::LazyGetter<Value> > getter(new TestGetter(c));
  {
    std::vector<edm::det_id_type> ids;
    ids.push_back(1);
    ids.push_back(3);

    edm::DetSetLazyVector<Value> lazyVector(getter, ids);
    CPPUNIT_ASSERT(lazyVector.size() == ids.size());

    dsv_type::const_iterator dsvItr = c.begin();
    for(edm::DetSetLazyVector<Value>::const_iterator it = lazyVector.begin(),
         itEnd = lazyVector.end();
         it != itEnd;
         ++it, ++dsvItr) {
      CPPUNIT_ASSERT(it->id == dsvItr->id);
      CPPUNIT_ASSERT(it->data.size() == dsvItr->data.size());
    }
  }

  {
    std::vector<edm::det_id_type> ids;
    ids.push_back(3);

    edm::DetSetLazyVector<Value> lazyVector(getter, ids);
    CPPUNIT_ASSERT(lazyVector.size() == ids.size());

    edm::DetSetLazyVector<Value>::const_iterator itRef = lazyVector.begin();
    for(std::vector<edm::det_id_type>::const_iterator itId = ids.begin(),
         itIdEnd = ids.end();
         itId != itIdEnd;
         ++itRef,++itId) {
      CPPUNIT_ASSERT(itRef->id == *itId);
      CPPUNIT_ASSERT(itRef->id == c.find(*itId)->id);
      CPPUNIT_ASSERT(itRef->data.size() == c.find(*itId)->data.size());
    }
  }
}

void
testDetSetLazyVector::checkFind()
{
  dsv_type c;
  detset    d3;
  Value v1(1.1);
  Value v2(2.2);
  d3.id = edm::det_id_type(3);
  d3.data.push_back(v1);
  d3.data.push_back(v2);
  c.insert(d3);
  detset    d1;
  Value v1a(4.1);
  Value v2a(3.2);
  d1.id = edm::det_id_type(1);
  d1.data.push_back(v1a);
  d1.data.push_back(v2a);
  c.insert(d1);
  c.post_insert();

  boost::shared_ptr<edm::dslv::LazyGetter<Value> > getter(new TestGetter(c));

  {
    std::vector<edm::det_id_type> ids;
    ids.push_back(1);
    ids.push_back(3);

    edm::DetSetLazyVector<Value> lazyVector(getter, ids);

    CPPUNIT_ASSERT(lazyVector.find(1)->id == c.find(1)->id);
    CPPUNIT_ASSERT(lazyVector.find(3)->id == c.find(3)->id);
    CPPUNIT_ASSERT(lazyVector.find(4) == lazyVector.end());
  }
}
