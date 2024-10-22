#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"

#include "DataFormats/Provenance/interface/CompactEventAuxiliaryVector.h"

#include "cppunit/extensions/HelperMacros.h"

class TestCompactEventAuxiliaryVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestCompactEventAuxiliaryVector);
  CPPUNIT_TEST(fillAndCompare);
  CPPUNIT_TEST_SUITE_END();

public:
  TestCompactEventAuxiliaryVector() {}
  ~TestCompactEventAuxiliaryVector() {}
  void setUp();
  void tearDown() {}

  void fillAndCompare();

private:
  edm::ProcessHistoryID phid_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCompactEventAuxiliaryVector);

void TestCompactEventAuxiliaryVector::setUp() {
  edm::ProcessConfiguration pc;
  auto processHistory = std::make_unique<edm::ProcessHistory>();
  edm::ProcessHistory& ph = *processHistory;
  processHistory->push_back(pc);
  phid_ = ph.id();
}

void TestCompactEventAuxiliaryVector::fillAndCompare() {
  std::vector<edm::EventAuxiliary> aux = {{{165121, 62, 23634374},
                                           "E403F8AD-A17F-E011-A42B-0022195E688C",
                                           edm::Timestamp((1305540804ULL << 32) + 10943758ULL),
                                           true,
                                           edm::EventAuxiliary::PhysicsTrigger,
                                           1995,
                                           0,
                                           16008401},
                                          {{165121, 62, 23643566},
                                           "A83A11AE-A17F-E011-A5F1-001EC9ADD952",
                                           edm::Timestamp((1305540804ULL << 32) + 20943758ULL),
                                           true,
                                           edm::EventAuxiliary::PhysicsTrigger,
                                           278,
                                           0,
                                           16014593},
                                          {{165121, 62, 23666070},
                                           "2E2618AE-A17F-E011-A82D-001EC9ADE1D1",
                                           edm::Timestamp((1305540805ULL << 32) + 10943758ULL),
                                           true,
                                           edm::EventAuxiliary::PhysicsTrigger,
                                           1981,
                                           0,
                                           16029687},
                                          {{165121, 62, 23666454},
                                           "F055F4AD-A17F-E011-9552-001517794E74",
                                           edm::Timestamp((1305540804ULL << 32) + 20943758ULL),
                                           true,
                                           edm::EventAuxiliary::PhysicsTrigger,
                                           611,
                                           0,
                                           16029970}};

  for (auto& a : aux) {
    a.setProcessHistoryID(phid_);
  }

  edm::CompactEventAuxiliaryVector caux;
  for (const auto& a : aux) {
    caux.push_back(a);
  }

  CPPUNIT_ASSERT(aux.size() == caux.size());

  auto j = caux.begin();
  for (auto i = aux.begin(); i != aux.end(); ++i, ++j) {
    CPPUNIT_ASSERT(edm::isSameEvent(*i, j->eventAuxiliary()));
  }

  auto i = aux.begin();
  j = caux.begin();
  i->id() = edm::EventID(165121, 62, 23634375);
  CPPUNIT_ASSERT(!edm::isSameEvent(*i, j->eventAuxiliary()));
}
