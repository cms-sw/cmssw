#include "cppunit/extensions/HelperMacros.h"
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationMap.h"

class testOneToOneAssociation : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testOneToOneAssociation);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll();
  void dummy();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testOneToOneAssociation);

void testOneToOneAssociation::checkAll() {
  typedef std::vector<int> CKey;
  typedef std::vector<double> CVal;
  typedef edm::AssociationMap<edm::OneToOne<CKey, CVal, unsigned char> > Assoc;
  Assoc v;
  CPPUNIT_ASSERT(v.empty());
  CPPUNIT_ASSERT(v.size() == 0);
}

// just check that some stuff compiles
void testOneToOneAssociation::dummy() {
  typedef std::vector<int> CKey;
  typedef std::vector<double> CVal;
  typedef edm::AssociationMap<edm::OneToOne<CKey, CVal, unsigned char> > Assoc;
  Assoc v;
  v.insert(edm::Ref<CKey>(), edm::Ref<CVal>());
  v.insert(Assoc::value_type(edm::Ref<CKey>(), edm::Ref<CVal>()));
  Assoc::const_iterator b = v.begin(), e = v.end();
  ++b;
  ++e;
  Assoc::const_iterator f = v.find(edm::Ref<CKey>());
  v.numberOfAssociations(edm::Ref<CKey>());
  const edm::Ref<CVal>& x = v[edm::Ref<CKey>()];
  x.id();
  ++f;
  int n = v.numberOfAssociations(edm::Ref<CKey>());
  ++n;
  std::cout <<"numberOfAssociations:"<<n<<std::endl;
  edm::Ref<Assoc> r;
  v[edm::Ref<CKey>()];
  v.erase(edm::Ref<CKey>());
  v.clear();
  CPPUNIT_ASSERT(v.size() == 0);
  v.post_insert();
}
