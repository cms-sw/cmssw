/*
 *  interval_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/30/05.
 *  Changed by Viji Sundararajan on 29-Jun-05.
 *
 */

#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cassert>

#include "FWCore/Framework/interface/PathsAndConsumesOfModules.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "cppunit/extensions/HelperMacros.h"

using ModuleDependsOnMap = std::map<std::string, std::vector<std::string>>;
using PathToModules = std::unordered_map<std::string, std::vector<std::string>>;

namespace {
  class PathsAndConsumesOfModulesForTest : public edm::PathsAndConsumesOfModulesBase {
  public:
    PathsAndConsumesOfModulesForTest(ModuleDependsOnMap const&, PathToModules const&);

  private:
    std::vector<std::string> const& doPaths() const final { return m_paths; }
    std::vector<std::string> const& doEndPaths() const final { return m_endPaths; }
    std::vector<edm::ModuleDescription const*> const& doAllModules() const final { return m_modules; }
    edm::ModuleDescription const* doModuleDescription(unsigned int moduleID) const final { return m_modules[moduleID]; }
    std::vector<edm::ModuleDescription const*> const& doModulesOnPath(unsigned int pathIndex) const final {
      return m_modulesOnPath[pathIndex];
    }
    std::vector<edm::ModuleDescription const*> const& doModulesOnEndPath(unsigned int endPathIndex) const final {
      return m_modulesOnEndPath[endPathIndex];
    }
    std::vector<edm::ModuleDescription const*> const& doModulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, edm::BranchType branchType) const final {
      return m_modulesWhoseProductsAreConsumedBy[moduleID];
    }
    std::vector<edm::ConsumesInfo> doConsumesInfo(unsigned int moduleID) const final {
      return m_moduleConsumesInfo[moduleID];
    }
    unsigned int doLargestModuleID() const final {
      if (m_modules.empty()) {
        return 0;
      }
      return m_modules.size() - 1;
    }

    std::vector<std::string> m_paths;
    std::vector<std::string> m_endPaths;
    std::vector<edm::ModuleDescription const*> m_modules;
    std::vector<std::vector<edm::ConsumesInfo>> m_moduleConsumesInfo;
    std::vector<std::vector<edm::ModuleDescription const*>> m_modulesOnPath;
    std::vector<std::vector<edm::ModuleDescription const*>> m_modulesOnEndPath;
    std::vector<std::vector<edm::ModuleDescription const*>> m_modulesWhoseProductsAreConsumedBy;
    std::vector<edm::ModuleDescription> m_cache;

    static unsigned int indexForModule(std::string const& iName,
                                       std::unordered_map<std::string, unsigned int>& modsToIndex,
                                       std::unordered_map<unsigned int, std::string>& indexToMods) {
      auto found = modsToIndex.find(iName);
      unsigned int fromIndex;
      if (found == modsToIndex.end()) {
        fromIndex = modsToIndex.size();
        modsToIndex.emplace(iName, fromIndex);
        indexToMods.emplace(fromIndex, iName);
      } else {
        fromIndex = found->second;
      }
      return fromIndex;
    }
  };
  PathsAndConsumesOfModulesForTest::PathsAndConsumesOfModulesForTest(ModuleDependsOnMap const& iModDeps,
                                                                     PathToModules const& iPaths) {
    //setup module indicies
    std::unordered_map<std::string, unsigned int> modsToIndex;
    std::unordered_map<unsigned int, std::string> indexToMods;

    const edm::ProcessConfiguration pc("TEST", edm::ParameterSetID{}, "CMSSW_x_y_z", "??");

    //In actual configuration building, the source is always assigned id==0
    m_cache.emplace_back(
        edm::ParameterSetID{}, "source", "source", &pc, indexForModule("source", modsToIndex, indexToMods));

    for (auto const& md : iModDeps) {
      auto const lastSize = modsToIndex.size();
      auto index = indexForModule(md.first, modsToIndex, indexToMods);
      if (index == lastSize) {
        m_cache.emplace_back(edm::ParameterSetID{}, md.first, md.first, &pc, index);
      }
    }
    m_paths.reserve(iPaths.size());
    for (auto const& pToM : iPaths) {
      m_paths.push_back(pToM.first);

      for (auto const& mod : pToM.second) {
        auto const lastSize = modsToIndex.size();
        unsigned int index = indexForModule(mod, modsToIndex, indexToMods);
        if (index == lastSize) {
          m_cache.emplace_back(edm::ParameterSetID{}, mod, mod, &pc, index);
        }
      }
    }
    for (auto const& md : iModDeps) {
      for (auto const& dep : md.second) {
        auto const lastSize = modsToIndex.size();
        auto index = indexForModule(dep, modsToIndex, indexToMods);
        if (index == lastSize) {
          m_cache.emplace_back(edm::ParameterSetID{}, dep, dep, &pc, index);
        }
      }
    }

    if (not iPaths.empty()) {
      auto indexForTriggerResults = indexForModule("TriggerResults", modsToIndex, indexToMods);
      for (auto const& pToM : iPaths) {
        auto index = indexForModule(pToM.first, modsToIndex, indexToMods);
        m_cache.emplace_back(edm::ParameterSetID{}, "PathStatusInserter", pToM.first, &pc, index);
      }
      m_cache.emplace_back(
          edm::ParameterSetID{}, "TriggerResultInserter", "TriggerResults", &pc, indexForTriggerResults);
    }

    m_modules.reserve(m_cache.size());
    for (auto const& desc : m_cache) {
      m_modules.push_back(&desc);
    }

    //do consumes
    edm::TypeID dummy;
    m_moduleConsumesInfo.resize(m_modules.size());
    m_modulesWhoseProductsAreConsumedBy.resize(m_modules.size());
    for (auto const& md : iModDeps) {
      auto moduleID = modsToIndex[md.first];
      auto& consumes = m_moduleConsumesInfo[moduleID];
      consumes.reserve(md.second.size());
      for (auto const& dep : md.second) {
        consumes.emplace_back(dummy, dep.c_str(), "", "TEST", edm::InEvent, edm::PRODUCT_TYPE, true, false);
        m_modulesWhoseProductsAreConsumedBy[moduleID].push_back(m_modules[modsToIndex[dep]]);
        //m_modulesWhoseProductsAreConsumedBy[modsToIndex[dep]].push_back(m_modules[moduleID]);
      }
    }

    m_modulesOnPath.reserve(m_paths.size());
    for (auto const& pToM : iPaths) {
      m_modulesOnPath.emplace_back();
      auto& newPath = m_modulesOnPath.back();
      newPath.reserve(pToM.second.size());
      for (auto const& mod : pToM.second) {
        newPath.push_back(m_modules[modsToIndex[mod]]);
      }
    }
  }
}  // namespace

class test_checkForModuleDependencyCorrectness : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_checkForModuleDependencyCorrectness);

  CPPUNIT_TEST(onePathNoCycleTest);
  CPPUNIT_TEST(onePathHasCycleTest);
  CPPUNIT_TEST(twoPathsNoCycleTest);
  CPPUNIT_TEST(twoPathsWithCycleTest);
  CPPUNIT_TEST(duplicateModuleOnPathTest);
  CPPUNIT_TEST(selfCycleTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void onePathNoCycleTest();
  void onePathHasCycleTest();

  void twoPathsNoCycleTest();
  void twoPathsWithCycleTest();

  void selfCycleTest();

  void duplicateModuleOnPathTest();

private:
  bool testCase(ModuleDependsOnMap const& iModDeps, PathToModules const& iPaths) const {
    PathsAndConsumesOfModulesForTest pAndC(iModDeps, iPaths);

    checkForModuleDependencyCorrectness(pAndC, false);
    return true;
  }
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(test_checkForModuleDependencyCorrectness);

void test_checkForModuleDependencyCorrectness::onePathNoCycleTest() {
  {
    ModuleDependsOnMap md = {{"C", {"B"}}, {"B", {"A"}}};
    PathToModules paths = {{"p", {"A", "B", "C"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    ModuleDependsOnMap md = {};
    PathToModules paths = {{"p", {"A", "B", "C"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    //Circular dependency but not connected to any path
    // NOTE: "A" is on the list since the ROOT of the
    // tree in the job is actually TriggerResults which
    // always connects to paths and end paths
    ModuleDependsOnMap md = {{"E", {"F"}}, {"F", {"G"}}, {"G", {"E"}}, {"A", {}}};
    PathToModules paths = {{"p", {"A", "B", "C"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }
}

void test_checkForModuleDependencyCorrectness::onePathHasCycleTest() {
  {
    ModuleDependsOnMap md = {{"C", {"B"}}, {"B", {"A"}}};
    {
      PathToModules paths = {{"p", {"B", "A", "C"}}};

      CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
    }
    {
      PathToModules paths = {{"p", {"B", "A"}}};

      CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
    }
    {
      PathToModules paths = {{"p", {"C", "A"}}};

      CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
    }
    {
      PathToModules paths = {{"p", {"C", "B"}}};

      CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
    }
  }

  {
    ModuleDependsOnMap md = {{"C", {"B"}}};
    {
      PathToModules paths = {{"p", {"C", "A", "B"}}};

      CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
    }
    {
      PathToModules paths = {{"p", {"A", "C", "B"}}};

      CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
    }
    {
      PathToModules paths = {{"p", {"C", "B", "A"}}};

      CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
    }
  }
}

void test_checkForModuleDependencyCorrectness::twoPathsNoCycleTest() {
  {
    ModuleDependsOnMap md = {{"C", {"B"}}};

    {
      PathToModules paths = {{"p1", {"A", "B", "C"}}, {"p2", {"A", "B", "C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }

    {
      //CDJ DEBUG THIS IS NOW FAILING
      PathToModules paths = {{"p1", {"A", "B", "C"}}, {"p2", {"B", "A", "C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"A", "B", "C"}}, {"p2", {"B", "C", "A"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"B", "A", "C"}}, {"p2", {"A", "B", "C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"B", "A", "C"}}, {"p2", {"B", "A", "C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"B", "A", "C"}}, {"p2", {"B", "C", "A"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"B", "C", "A"}}, {"p2", {"A", "B", "C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"B", "C", "A"}}, {"p2", {"B", "A", "C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"B", "C", "A"}}, {"p2", {"B", "C", "A"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
  }

  {
    //Test all possible 3 module combinations
    ModuleDependsOnMap md = {};
    std::vector<std::string> moduleName = {"A", "B", "C"};
    std::vector<std::string> pathModules;
    for (unsigned int i = 0; i < 3; ++i) {
      pathModules.push_back(moduleName[i]);
      for (unsigned int j = 0; j < 3; ++j) {
        if (j == i) {
          continue;
        }
        pathModules.push_back(moduleName[j]);
        for (unsigned int k = 0; k < 3; ++k) {
          if (j == k or i == k) {
            continue;
          }
          pathModules.push_back(moduleName[k]);

          std::vector<std::string> path2Modules;
          for (unsigned int ii = 0; ii < 3; ++ii) {
            path2Modules.push_back(moduleName[ii]);
            for (unsigned int jj = 0; jj < 3; ++jj) {
              if (jj == ii) {
                continue;
              }
              path2Modules.push_back(moduleName[jj]);
              for (unsigned int kk = 0; kk < 3; ++kk) {
                if (jj == kk or ii == kk) {
                  continue;
                }
                path2Modules.push_back(moduleName[kk]);
                PathToModules paths;
                paths["p1"] = pathModules;
                paths["p2"] = path2Modules;
                CPPUNIT_ASSERT(testCase(md, paths));
                path2Modules.pop_back();
              }
              path2Modules.pop_back();
            }
            path2Modules.pop_back();
          }

          pathModules.pop_back();
        }
        pathModules.pop_back();
      }
      pathModules.pop_back();
    }
  }

  {
    ModuleDependsOnMap md = {{"C", {"B"}}, {"B", {"A"}}, {"D", {"C"}}};

    {
      PathToModules paths = {{"p1", {"A", "C"}}, {"p2", {"B", "D"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }

    {
      PathToModules paths = {{"p1", {"A", "D"}}, {"p2", {"B", "C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }

    {
      PathToModules paths = {{"p1", {"A"}}, {"p2", {"C", "D"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"A"}}, {"p2", {"D"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"A"}}, {"p2", {"C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }

    {
      PathToModules paths = {{"p1", {"B"}}, {"p2", {"D"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }

    {
      PathToModules paths = {{"p1", {"B"}}, {"p2", {"C"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"C"}}, {"p2", {"D"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }

    {
      PathToModules paths = {{"p1", {"A", "C"}}, {"p2", {"A", "D"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
    {
      PathToModules paths = {{"p1", {"A", "C"}}, {"p2", {"A", "D"}}};

      CPPUNIT_ASSERT(testCase(md, paths));
    }
  }

  {
    ModuleDependsOnMap md = {{"B", {"C"}}, {"D", {"A"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    ModuleDependsOnMap md = {{"B", {"E"}}, {"E", {"C"}}, {"D", {"F"}}, {"F", {"A"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    ModuleDependsOnMap md = {{"B", {"E"}}, {"E", {"C"}}, {"D", {"E"}}, {"E", {"A"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    ModuleDependsOnMap md = {{"B", {"E"}}, {"C", {"E"}}, {"D", {"E"}}, {"A", {"E"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    //Simplified schedule which was failing
    ModuleDependsOnMap md = {{"H", {"A"}}};
    PathToModules paths = {{"reco", {"I", "A", "H", "G", "F", "E"}}, {"val", {"E", "D", "C", "B", "A"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    //Simplified schedule which was failing
    ModuleDependsOnMap md = {{"B", {"X"}}, {"Y", {"Z"}}, {"Z", {"A"}}};
    PathToModules paths = {{"p1", {"X", "B", "A"}}, {"p2", {"A", "Z", "?", "Y", "X"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }
  {
    //Simplified schedule which was failing
    ModuleDependsOnMap md = {{"B", {"X"}}, {"Y", {"Z"}}, {"Z", {"A"}}, {"?", {}}, {"A", {}}, {"X", {}}};
    PathToModules paths = {{"p1", {"X", "B", "A"}}, {"p2", {"A", "Z", "?", "Y", "X"}}};
    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    //Simplified schedule which was failing
    // The data dependency for 'D" can be met
    // by the order of modules on path p2
    ModuleDependsOnMap md = {{"A_TR", {"zEP1", "zEP2"}}, {"D", {"B"}}, {"E", {"D"}}, {"zSEP3", {"A_TR"}}};
    PathToModules paths = {{"p1", {"E", "F", "zEP1"}}, {"p2", {"B", "C", "D", "zEP2"}}, {"p3", {"zSEP3", "B", "zEP3"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    //The same sequence of modules appear on a Path and EndPath
    // Check that the framework does not get confused when jumping
    // from one path to the other path just because of the
    // TriggerResults connection.

    ModuleDependsOnMap md = {{"A_TR", {"zEP1"}},
                             {"A", {"B"}},
                             {"B", {"H"}},
                             {"C", {}},
                             {"D", {}},
                             {"E", {}},
                             {"G", {"D"}},
                             {"H", {"D"}},
                             {"zEP1", {}},
                             {"zSEP2", {"A_TR"}}};
    PathToModules paths = {{"p2", {"D", "G", "H", "B", "C", "zEP1"}},
                           {"p3", {"A"}},  //Needed to make graph search start here
                           {"p1", {"zSEP2", "E", "D", "G", "H", "B", "C"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    //Have a module which can not be run initially be needed by two other modules
    ModuleDependsOnMap md = {{"out", {"A", "B"}}, {"A", {"D"}}, {"B", {"D"}}};
    PathToModules paths = {{"p1", {"filter", "D"}}, {"p2", {"out"}}};
    CPPUNIT_ASSERT(testCase(md, paths));
  }
  {
    //like above, but with path names reversed
    ModuleDependsOnMap md = {{"out", {"A", "B"}}, {"A", {"D"}}, {"B", {"D"}}};
    PathToModules paths = {{"p1", {"out"}}, {"p2", {"filter", "D"}}};
    CPPUNIT_ASSERT(testCase(md, paths));
  }

  {
    //Have a module which can not be run initially be needed by two other modules
    ModuleDependsOnMap md = {{"out", {"A", "B"}}, {"A", {"D"}}, {"B", {"D"}}, {"D", {"E"}}};
    PathToModules paths = {{"p1", {"filter", "E"}}, {"p2", {"out"}}};
    CPPUNIT_ASSERT(testCase(md, paths));
  }
  {
    //like above, but with path names reversed
    ModuleDependsOnMap md = {{"out", {"A", "B"}}, {"A", {"D"}}, {"B", {"D"}}, {"D", {"E"}}};
    PathToModules paths = {{"p1", {"out"}}, {"p2", {"filter", "E"}}};
    CPPUNIT_ASSERT(testCase(md, paths));
  }
}

void test_checkForModuleDependencyCorrectness::twoPathsWithCycleTest() {
  {
    ModuleDependsOnMap md = {{"C", {"B"}}, {"A", {"D"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    //Add additional dependencies to test that they are ignored
    ModuleDependsOnMap md = {{"C", {"E", "F", "G", "B"}}, {"A", {"D"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = {{"C", {"B"}}, {"A", {"D"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D", "A"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = {{"C", {"B"}}, {"A", {"D"}}, {"B", {"A"}}, {"D", {"C"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = {{"E", {"B"}}, {"C", {"E"}}, {"A", {"D"}}};
    PathToModules paths = {{"p1", {"A", "B"}}, {"p2", {"C", "D"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = {{"A", {"B"}}, {"B", {"C"}}, {"C", {"D"}}, {"D", {"B"}}};
    PathToModules paths = {{"p1", {"A", "EP"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    //Simplified schedule which was failing
    ModuleDependsOnMap md = {{"B", {"X"}}, {"Y", {"Z"}}, {"Z", {"A"}}, {"?", {}}, {"A", {}}, {"X", {}}};
    //NOTE: p1 is inconsistent but with p2 it would be runnable.
    PathToModules paths = {{"p1", {"B", "A", "X"}}, {"p2", {"A", "Z", "?", "Y", "X"}}};
    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = {{"A_TR", {"EP1", "EP2"}},  //assigned aTR==0, EP1==1, EP2==2
                             {"C", {"A"}},
                             {"D", {"A"}},
                             {"E", {"D"}},
                             {"BP", {"A_TR"}}};

    PathToModules paths = {{"p1", {"A", "B", "C", "EP1"}}, {"p2", {"E", "EP2"}}, {"ep", {"BP", "A", "D"}}};
    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    // The data dependency for 'D" can be met
    // by the order of modules on path p2
    // but NOT by path3
    ModuleDependsOnMap md = {{"A_TR", {"zEP1", "zEP2", "zEP3"}}, {"D", {"B"}}, {"E", {"D"}}, {"zSEP4", {"A_TR"}}};
    PathToModules paths = {{"p1", {"E", "F", "zEP1"}},
                           {"p2", {"Filter", "B", "C", "D", "zEP2"}},
                           {"p3", {"C", "D", "zEP3"}},
                           {"p4", {"zSEP4", "B", "zEP4"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    // The data dependency for 'D" can be met
    // by the order of modules on path p2
    // but NOT by path3
    ModuleDependsOnMap md = {{"A_TR", {"zEP1", "zEP2", "zEP3"}}, {"D", {"B"}}, {"E", {"D"}}, {"zSEP4", {"A_TR"}}};
    PathToModules paths = {{"p1", {"E", "F", "zEP1"}},
                           {"p2", {"Filter", "B", "D", "zEP2"}},
                           {"p3", {"C", "D", "zEP3"}},
                           {"p4", {"zSEP4", "B", "zEP4"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    // The data dependency for 'D" can be met
    // by the order of modules on path p2
    // but NOT by path3
    ModuleDependsOnMap md = {{"A_TR", {"zEP1", "zEP2"}}, {"B", {}}, {"zFilter", {"A_TR"}}};
    PathToModules paths = {{"p1", {"zFilter", "B", "zEP1"}}, {"p2", {"zFilter", "B", "zEP2"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = {{"B", {"A"}}, {"C", {"B"}}, {"cFilter", {"C"}}};
    PathToModules paths = {{"p1", {"C", "cFilter", "D", "E", "F", "A", "B"}}, {"p2", {"oFilter", "D", "F", "B"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }
}

void test_checkForModuleDependencyCorrectness::selfCycleTest() {
  {
    ModuleDependsOnMap md = {{"A", {"A"}}};
    PathToModules paths = {{"p", {"A"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }
  {
    ModuleDependsOnMap md = {{"A", {"A"}}, {"B", {"A"}}};
    PathToModules paths = {{"p", {"B"}}};

    CPPUNIT_ASSERT_THROW(testCase(md, paths), cms::Exception);
  }
}

void test_checkForModuleDependencyCorrectness::duplicateModuleOnPathTest() {
  {
    ModuleDependsOnMap md = {{"C", {"B"}}, {"B", {"A"}}};
    PathToModules paths = {{"p", {"A", "B", "C", "A"}}};

    CPPUNIT_ASSERT(testCase(md, paths));
  }
}
