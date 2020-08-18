#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/TableView.h"
#include "FWCore/SOA/interface/Column.h"

#include <cppunit/extensions/HelperMacros.h>

class testTableFilling : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTableFilling);

  CPPUNIT_TEST(soaDeclareDefaultTest1);
  CPPUNIT_TEST(soaDeclareDefaultTest2);
  CPPUNIT_TEST(soaDeclareDefaultTest3);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void soaDeclareDefaultTest1();
  void soaDeclareDefaultTest2();
  void soaDeclareDefaultTest3();
};

namespace edm::soa {

  namespace col {
    SOA_DECLARE_COLUMN(Eta, float, "eta");
    SOA_DECLARE_COLUMN(Phi, float, "phi");
    SOA_DECLARE_COLUMN(Energy, float, "energy");
    SOA_DECLARE_COLUMN(ID, int, "id");
    SOA_DECLARE_COLUMN(Label, std::string, "label");

    SOA_DECLARE_COLUMN(Px, double, "p_x");
    SOA_DECLARE_COLUMN(Py, double, "p_y");
    SOA_DECLARE_COLUMN(Pz, double, "p_z");
  }  // namespace col

  // The SOA_DECLARE_DEFAULT macro should be used in the edm::soa namespace
  SOA_DECLARE_DEFAULT(col::Eta, eta());
  SOA_DECLARE_DEFAULT(col::Phi, phi());
  SOA_DECLARE_DEFAULT(col::Energy, energy());
  SOA_DECLARE_DEFAULT(col::ID, id());
  SOA_DECLARE_DEFAULT(col::Label, label());
  SOA_DECLARE_DEFAULT(col::Px, px());
  SOA_DECLARE_DEFAULT(col::Py, py());
  SOA_DECLARE_DEFAULT(col::Pz, pz());

  using EtaPhiTable = Table<col::Eta, col::Phi>;
  using EtaPhiTableView = ViewFromTable_t<EtaPhiTable>;
}  // namespace edm::soa

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTableFilling);

namespace {
  template <class Object>
  void validateEtaPhiTable(edm::soa::EtaPhiTableView table, std::vector<Object> const& objects) {
    using namespace edm::soa;
    int iRow = 0;
    for (auto const& row : table) {
      CPPUNIT_ASSERT(row.get<col::Eta>() == objects[iRow].eta_);
      CPPUNIT_ASSERT(row.get<col::Phi>() == objects[iRow].phi_);
      ++iRow;
    }
  }
}  // namespace

struct JetType1 {
  // jet type that is fully compatible with the
  // SOA_DECLARE_DEFAULT-generated value_for_column functions
  auto eta() const { return eta_; }
  auto phi() const { return phi_; }

  const float eta_;
  const float phi_;
};

void testTableFilling::soaDeclareDefaultTest1() {
  using namespace edm::soa;

  std::vector<JetType1> jets = {{1., 3.14}, {2., 0.}, {4., 1.3}};
  std::vector<std::string> labels = {{"jet0", "jet1", "jet2"}};

  EtaPhiTable table(jets);

  validateEtaPhiTable(table, jets);
}

struct JetType2 {
  // jet type that is NOT compatible with the SOA_DECLARE_DEFAULT-generated
  // value_for_column functions...
  auto eta() const { return eta_; }

  const float eta_;
  const float phi_;
};

// ...so we need to write our own value_for_column function for JetType2
double value_for_column(JetType2 const& iJ, edm::soa::col::Phi*) { return iJ.phi_; }

void testTableFilling::soaDeclareDefaultTest2() {
  using namespace edm::soa;

  std::vector<JetType2> jets = {{1., 3.14}, {2., 0.}, {4., 1.3}};
  std::vector<std::string> labels = {{"jet0", "jet1", "jet2"}};

  EtaPhiTable table(jets);

  validateEtaPhiTable(table, jets);
}

struct JetType3 {
  // jet type that is compatible with the SOA_DECLARE_DEFAULT-generated
  // value_for_column functions, but it would not give the right result.
  // This is to check that the custom value_for_column defined below is
  // prioritized.
  auto eta() const { return eta_; }
  auto phi() const { return 0.f; }

  const float eta_;
  const float phi_;
};

double value_for_column(JetType3 const& iJ, edm::soa::col::Phi*) { return iJ.phi_; }

void testTableFilling::soaDeclareDefaultTest3() {
  using namespace edm::soa;

  std::vector<JetType3> jets = {{1., 3.14}, {2., 0.}, {4., 1.3}};
  std::vector<std::string> labels = {{"jet0", "jet1", "jet2"}};

  EtaPhiTable table(jets);

  validateEtaPhiTable(table, jets);
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
