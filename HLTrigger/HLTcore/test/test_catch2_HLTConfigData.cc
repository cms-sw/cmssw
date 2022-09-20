#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "HLTrigger/HLTcore/interface/HLTConfigData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>
#include <map>

namespace {
  // build PSet of generic module
  edm::ParameterSet buildModulePSet(std::string const& iLabel, std::string const& iType, std::string const& iEDMType) {
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@module_label", iLabel);
    pset.addParameter<std::string>("@module_type", iType);
    pset.addParameter<std::string>("@module_edm_type", iEDMType);
    return pset;
  }

  // build PSet of PrescaleService module
  edm::ParameterSet buildPrescaleServicePSet(std::string const& iLabel,
                                             std::string const& iType,
                                             std::string const& iEDMType,
                                             std::vector<std::string> const& labels,
                                             std::map<std::string, std::vector<unsigned int>> const& prescaleTable) {
    auto pset = buildModulePSet(iLabel, iType, iEDMType);
    pset.addParameter("lvl1Labels", labels);
    std::vector<edm::ParameterSet> psTable;
    psTable.reserve(psTable.size());
    for (auto const& [key, val] : prescaleTable) {
      REQUIRE(labels.size() == val.size());
      edm::ParameterSet psEntry;
      psEntry.addParameter<std::string>("pathName", key);
      psEntry.addParameter<std::vector<unsigned int>>("prescales", val);
      psTable.emplace_back(psEntry);
    }
    pset.addParameter<std::vector<edm::ParameterSet>>("prescaleTable", psTable);
    return pset;
  }

}  // namespace

TEST_CASE("Test HLTConfigData", "[HLTConfigData]") {
  SECTION("TriggerPaths") {
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@process_name", "TEST");
    const std::vector<std::string> names = {{"b1"}, {"b2"}, {"a1"}, {"z5"}};
    {
      edm::ParameterSet tpset;
      tpset.addParameter<std::vector<std::string>>("@trigger_paths", names);
      pset.addParameter<edm::ParameterSet>("@trigger_paths", tpset);
    }
    pset.addParameter<std::vector<std::string>>("b1", {{"f1"}, {"f2"}});
    pset.addParameter<std::vector<std::string>>("b2", {{"f3"}, {"#"}, {"c1"}, {"c2"}, {"@"}});
    pset.addParameter<std::vector<std::string>>("a1", {{"f1"}, {"f4"}});
    pset.addParameter<std::vector<std::string>>("z5", {{"f5"}});

    pset.addParameter<edm::ParameterSet>("f1", buildModulePSet("f1", "F1Filter", "EDFilter"));
    pset.addParameter<edm::ParameterSet>("f2", buildModulePSet("f2", "F2Filter", "EDFilter"));
    pset.addParameter<edm::ParameterSet>("f3", buildModulePSet("f3", "F3Filter", "EDFilter"));
    pset.addParameter<edm::ParameterSet>("f4", buildModulePSet("f4", "F4Filter", "EDFilter"));
    pset.addParameter<edm::ParameterSet>("f5", buildModulePSet("f5", "F5Filter", "EDFilter"));

    pset.addParameter<edm::ParameterSet>("c1", buildModulePSet("c1", "CProducer", "EDProducer"));
    pset.addParameter<edm::ParameterSet>("c2", buildModulePSet("c2", "CProducer", "EDProducer"));

    pset.addParameter<edm::ParameterSet>(
        "PrescaleService",
        buildPrescaleServicePSet("PrescaleService",
                                 "PrescaleService",
                                 "Service",
                                 {"col0", "col1", "col2", "col3", "col4"},
                                 {{"b1", {45, 12, 1, 0, 1000}}, {"b2", {12000, 2, 0, 7, 0}}}));

    pset.registerIt();

    HLTConfigData cd(&pset);

    SECTION("check paths") {
      REQUIRE(cd.size() == 4);
      REQUIRE(cd.triggerName(0) == names[0]);
      REQUIRE(cd.triggerName(1) == names[1]);
      REQUIRE(cd.triggerName(2) == names[2]);
      REQUIRE(cd.triggerName(3) == names[3]);
      //cd.dump("Triggers");
    }

    SECTION("check modules on paths") {
      REQUIRE(cd.size(0) == 2);
      REQUIRE(cd.moduleLabel(0, 0) == "f1");
      REQUIRE(cd.moduleLabel(0, 1) == "f2");

      REQUIRE(cd.size(1) == 3);
      REQUIRE(cd.moduleLabel(1, 0) == "f3");
      REQUIRE(cd.moduleLabel(1, 1) == "c1");
      REQUIRE(cd.moduleLabel(1, 2) == "c2");

      REQUIRE(cd.size(2) == 2);
      REQUIRE(cd.moduleLabel(2, 0) == "f1");
      REQUIRE(cd.moduleLabel(2, 1) == "f4");

      REQUIRE(cd.size(3) == 1);
      REQUIRE(cd.moduleLabel(3, 0) == "f5");
      //cd.dump("Modules");
    }

    SECTION("check prescales") {
      // get prescale table reading values as double and FractionalPrescale
      auto const& psTableDouble = cd.prescaleTable<double>();
      auto const& psTableFractl = cd.prescaleTable<FractionalPrescale>();
      REQUIRE(psTableDouble.size() == psTableFractl.size());
      for (auto const& [key, vec_d] : psTableDouble) {
        auto const& vec_f = psTableFractl.at(key);
        REQUIRE(vec_d.size() == vec_f.size());
        for (size_t idx = 0; idx < vec_d.size(); ++idx) {
          auto const& val_d = vec_d[idx];
          auto const& val_f = vec_f[idx];
          // conversion of prescale value to unsigned int
          unsigned int const val_u = vec_d[idx];
          // equal-to comparison of double-or-FractionalPrescale to unsigned int must succeed:
          // HLT does not yet fully support non-integer HLT prescales (example: PrescaleService),
          // but the HLTConfigData utility (interface to downstream clients) provides access
          // to HLT prescales only via types 'double' and 'FractionalPrescale',
          // in anticipation of when HLT will fully support the use of non-integer prescales
          REQUIRE(val_d == val_u);
          REQUIRE(val_f == val_u);
        }
      }
      //cd.dump("PrescaleTable");
    }
  }
}
