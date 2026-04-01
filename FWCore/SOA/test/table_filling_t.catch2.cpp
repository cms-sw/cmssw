#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/TableView.h"
#include "FWCore/SOA/interface/Column.h"

#include "catch2/catch_all.hpp"

namespace ts {

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

  using EtaPhiTable = edm::soa::Table<ts::col::Eta, ts::col::Phi>;
  using EtaPhiTableView = edm::soa::ViewFromTable_t<EtaPhiTable>;
}  // namespace ts

namespace edm::soa {

  // The SOA_DECLARE_DEFAULT macro should be used in the edm::soa namespace
  SOA_DECLARE_DEFAULT(ts::col::Eta, eta());
  SOA_DECLARE_DEFAULT(ts::col::Phi, phi());
  SOA_DECLARE_DEFAULT(ts::col::Energy, energy());
  SOA_DECLARE_DEFAULT(ts::col::ID, id());
  SOA_DECLARE_DEFAULT(ts::col::Label, label());
  SOA_DECLARE_DEFAULT(ts::col::Px, px());
  SOA_DECLARE_DEFAULT(ts::col::Py, py());
  SOA_DECLARE_DEFAULT(ts::col::Pz, pz());

}  // namespace edm::soa

namespace {
  template <class Object>
  void validateEtaPhiTable(ts::EtaPhiTableView table, std::vector<Object> const& objects) {
    int iRow = 0;
    for (auto const& row : table) {
      REQUIRE(row.get<ts::col::Eta>() == objects[iRow].eta_);
      REQUIRE(row.get<ts::col::Phi>() == objects[iRow].phi_);
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

struct JetType2 {
  // jet type that is NOT compatible with the SOA_DECLARE_DEFAULT-generated
  // value_for_column functions...
  auto eta() const { return eta_; }

  const float eta_;
  const float phi_;
};

// ...so we need to write our own value_for_column function for JetType2
double value_for_column(JetType2 const& iJ, ts::col::Phi*) { return iJ.phi_; }

namespace ts::reco {

  struct JetType3 {
    auto eta() const { return eta_; }

    const float eta_;
    const float phi_;
  };

  double value_for_column(ts::reco::JetType3 const& iJ, ts::col::Phi*) { return iJ.phi_; }
}  // namespace ts::reco

struct JetType4 {
  // jet type that is compatible with the SOA_DECLARE_DEFAULT-generated
  // value_for_column functions, but it would not give the right result.
  // This is to check that the custom value_for_column defined below is
  // prioritized.
  auto eta() const { return eta_; }
  auto phi() const { return 0.f; }

  const float eta_;
  const float phi_;
};

double value_for_column(JetType4 const& iJ, ts::col::Phi*) { return iJ.phi_; }

namespace ts::reco {

  struct JetType5 {
    // jet type that is compatible with the SOA_DECLARE_DEFAULT-generated
    // value_for_column functions, but it would not give the right result.
    // This is to check that the custom value_for_column defined below is
    // prioritized.
    auto eta() const { return eta_; }
    auto phi() const { return 0.f; }

    const float eta_;
    const float phi_;
  };

  double value_for_column(ts::reco::JetType5 const& iJ, ts::col::Phi*) { return iJ.phi_; }

}  // namespace ts::reco

TEST_CASE("TableFilling", "[SOA]") {
  SECTION("soaDeclareDefaultTest1") {
    std::vector<JetType1> jets = {{1., 3.14}, {2., 0.}, {4., 1.3}};
    std::vector<std::string> labels = {{"jet0", "jet1", "jet2"}};

    ts::EtaPhiTable table(jets);

    validateEtaPhiTable(table, jets);
  }

  SECTION("soaDeclareDefaultTest2") {
    std::vector<JetType2> jets = {{1., 3.14}, {2., 0.}, {4., 1.3}};
    std::vector<std::string> labels = {{"jet0", "jet1", "jet2"}};

    ts::EtaPhiTable table(jets);

    validateEtaPhiTable(table, jets);
  }

  SECTION("soaDeclareDefaultTest3") {
    std::vector<ts::reco::JetType3> jets = {{1., 3.14}, {2., 0.}, {4., 1.3}};
    std::vector<std::string> labels = {{"jet0", "jet1", "jet2"}};

    ts::EtaPhiTable table(jets);

    validateEtaPhiTable(table, jets);
  }

  SECTION("soaDeclareDefaultTest4") {
    std::vector<JetType4> jets = {{1., 3.14}, {2., 0.}, {4., 1.3}};
    std::vector<std::string> labels = {{"jet0", "jet1", "jet2"}};

    ts::EtaPhiTable table(jets);

    validateEtaPhiTable(table, jets);
  }

  SECTION("soaDeclareDefaultTest5") {
    std::vector<ts::reco::JetType5> jets = {{1., 3.14}, {2., 0.}, {4., 1.3}};
    std::vector<std::string> labels = {{"jet0", "jet1", "jet2"}};

    ts::EtaPhiTable table(jets);

    validateEtaPhiTable(table, jets);
  }
}
