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

#include "FWCore/Framework/src/throwIfImproperDependencies.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "cppunit/extensions/HelperMacros.h"

class test_throwIfImproperDependencies : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_throwIfImproperDependencies);

  CPPUNIT_TEST(onePathNoCycleTest);
  CPPUNIT_TEST(onePathHasCycleTest);
  CPPUNIT_TEST(twoPathsNoCycleTest);
  CPPUNIT_TEST(twoPathsWithCycleTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void onePathNoCycleTest();
  void onePathHasCycleTest();

  void twoPathsNoCycleTest();
  void twoPathsWithCycleTest();

  using ModuleDependsOnMap = std::map<std::string, std::vector<std::string>>;
  using PathToModules = std::unordered_map<std::string, std::vector<std::string>>;

private:
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

  bool testCase(ModuleDependsOnMap const& iModDeps, PathToModules const& iPaths) const {
    using namespace edm::graph;

    EdgeToPathMap edgeToPathMap;

    std::unordered_map<std::string, unsigned int> modsToIndex;
    std::unordered_map<unsigned int, std::string> indexToMods;

    //We have an artificial case to be the root of the graph
    const std::string kFinishedProcessing("FinishedProcessing");
    const unsigned int kFinishedProcessingIndex{0};
    modsToIndex.emplace(kFinishedProcessing, kFinishedProcessingIndex);
    indexToMods.emplace(kFinishedProcessingIndex, kFinishedProcessing);

    //Setup the module to index map by using all module names used in both containers
    //Start with keys from the module dependency to allow control of the numbering scheme
    for (auto const& md : iModDeps) {
      indexForModule(md.first, modsToIndex, indexToMods);
    }

    std::vector<std::vector<unsigned int>> pathIndexToModuleIndexOrder(iPaths.size());

    //Need to be able to quickly look up which paths a module appears on
    std::unordered_map<unsigned int, std::vector<unsigned int>> moduleIndexToPathIndex;

    std::vector<std::string> pathNames;
    std::unordered_map<std::string, unsigned int> pathToIndexMap;
    for (auto const& path : iPaths) {
      unsigned int lastModuleIndex = kInvalidIndex;
      pathNames.push_back(path.first);
      unsigned int pathIndex = pathToIndexMap.size();
      auto& pathOrder = pathIndexToModuleIndexOrder[pathIndex];
      pathToIndexMap.emplace(path.first, pathIndex);
      for (auto const& mod : path.second) {
        unsigned int index = indexForModule(mod, modsToIndex, indexToMods);
        pathOrder.push_back(index);
        moduleIndexToPathIndex[index].push_back(pathIndex);

        if (lastModuleIndex != kInvalidIndex) {
          edgeToPathMap[std::make_pair(index, lastModuleIndex)].push_back(pathIndex);
        }
        lastModuleIndex = index;
      }
      pathOrder.push_back(kFinishedProcessingIndex);
      if (lastModuleIndex != kInvalidIndex) {
        edgeToPathMap[std::make_pair(kFinishedProcessingIndex, lastModuleIndex)].push_back(pathIndex);
      }
    }

    for (auto const& md : iModDeps) {
      unsigned int fromIndex = indexForModule(md.first, modsToIndex, indexToMods);
      for (auto const& dependsOn : md.second) {
        unsigned int toIndex = indexForModule(dependsOn, modsToIndex, indexToMods);

        //see if all paths containing this module also contain the dependent module earlier in the path
        // if it does, then treat this only as a path dependency and not a data dependency as this
        // simplifies the circular dependency checking logic

        auto itPathsFound = moduleIndexToPathIndex.find(fromIndex);
        bool keepDataDependency = true;
        if (itPathsFound != moduleIndexToPathIndex.end() and
            moduleIndexToPathIndex.find(toIndex) != moduleIndexToPathIndex.end()) {
          keepDataDependency = false;
          for (auto const pathIndex : itPathsFound->second) {
            for (auto idToCheck : pathIndexToModuleIndexOrder[pathIndex]) {
              if (idToCheck == toIndex) {
                //found dependent module first so check next path
                break;
              }
              if (idToCheck == fromIndex) {
                //did not find dependent module earlier on path so
                // must keep data dependency
                keepDataDependency = true;
                break;
              }
            }
            if (keepDataDependency) {
              break;
            }
          }
        }
        if (keepDataDependency) {
          edgeToPathMap[std::make_pair(fromIndex, toIndex)].push_back(kDataDependencyIndex);
        }
      }
    }

    throwIfImproperDependencies(edgeToPathMap, pathIndexToModuleIndexOrder, pathNames, indexToMods);

    return true;
  }
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(test_throwIfImproperDependencies);

void test_throwIfImproperDependencies::onePathNoCycleTest() {
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

void test_throwIfImproperDependencies::onePathHasCycleTest() {
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

void test_throwIfImproperDependencies::twoPathsNoCycleTest() {
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
}

void test_throwIfImproperDependencies::twoPathsWithCycleTest() {
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
}
