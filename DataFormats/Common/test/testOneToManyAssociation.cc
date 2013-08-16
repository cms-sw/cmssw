#include "cppunit/extensions/HelperMacros.h"
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationMap.h"

class testOneToManyAssociation : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testOneToManyAssociation);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll();
  void dummy();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testOneToManyAssociation);

void testOneToManyAssociation::checkAll() {
  typedef std::vector<int> CKey;
  typedef std::vector<double> CVal;
  typedef edm::AssociationMap<edm::OneToMany<CKey, CVal, unsigned char> > Assoc;
  Assoc v;
  CPPUNIT_ASSERT(v.empty());
  CPPUNIT_ASSERT(v.size() == 0);
}

// just check that some stuff compiles
void testOneToManyAssociation::dummy() {
  typedef std::vector<int> CKey;
  typedef std::vector<double> CVal;
  {
    typedef edm::AssociationMap<edm::OneToMany<CKey, CVal, unsigned char> > Assoc;
    Assoc v;
    v.insert(edm::Ref<CKey>(), edm::Ref<CVal>());
    v.insert(Assoc::value_type(edm::Ref<CKey>(), edm::RefVector<CVal>()));
    Assoc::const_iterator b = v.begin(), e = v.end();
    ++b; ++e;
    Assoc::const_iterator f = v.find(edm::Ref<CKey>());
    v.numberOfAssociations(edm::Ref<CKey>());
    const edm::RefVector<CVal> & x = v[edm::Ref<CKey>()]; x.size();
    ++f;
    int n = v.numberOfAssociations(edm::Ref<CKey>());
    ++n;
    edm::Ref<Assoc> r;
    v[edm::Ref<CKey>()];
    v.erase(edm::Ref<CKey>());
    v.clear();
    CPPUNIT_ASSERT(v.size() == 0);
    v.post_insert();
  }
  {
    typedef edm::AssociationMap<edm::OneToManyWithQuality<CKey, CVal, double, unsigned char> > Assoc;
    Assoc v;
    v.insert(edm::Ref<CKey>(), std::make_pair(edm::Ref<CVal>(), 3.14));
    Assoc::const_iterator b = v.begin(), e = v.end();
    ++b; ++e;
    Assoc::const_iterator f = v.find(edm::Ref<CKey>());
    v.numberOfAssociations(edm::Ref<CKey>());
    const std::vector<std::pair<edm::Ref<CVal>, double> > & x = v[edm::Ref<CKey>()]; x.size();
    ++f;
    int n = v.numberOfAssociations(edm::Ref<CKey>());
    ++n;
    edm::Ref<Assoc> r;
    v[edm::Ref<CKey>()];
    v.erase(edm::Ref<CKey>());
    v.clear();
    CPPUNIT_ASSERT(v.size() == 0);
    v.post_insert();
  }
}
