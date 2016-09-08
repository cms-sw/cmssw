/*
 *  interval_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/30/05.
 *  Changed by Viji Sundararajan on 29-Jun-05.
 *
 */

#include "FWCore/Framework/src/throwIfImproperDependencies.h"
#include "cppunit/extensions/HelperMacros.h"
#include <unordered_map>

#include "FWCore/Utilities/interface/Exception.h"

class test_throwIfImproperDependencies: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(test_throwIfImproperDependencies);

  CPPUNIT_TEST(onePathNoCycleTest);
  CPPUNIT_TEST(onePathHasCycleTest);
  CPPUNIT_TEST(twoPathsNoCycleTest);
  CPPUNIT_TEST(twoPathsWithCycleTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void onePathNoCycleTest();
  void onePathHasCycleTest();

  void twoPathsNoCycleTest();
  void twoPathsWithCycleTest();

  using ModuleDependsOnMap = std::unordered_map<std::string, std::vector<std::string>>;
  using PathToModules = std::unordered_map<std::string, std::vector<std::string>>;
  
private:
  bool testCase( ModuleDependsOnMap const& iModDeps, PathToModules const& iPaths) {
    using namespace edm::graph;
    
    EdgeToPathMap edgeToPathMap;
    
    std::map<std::string, unsigned int> modsToIndex;
    
    //Setup the module to index map by using all module names used in both containers
    for(auto const& md : iModDeps) {
      auto found = modsToIndex.find(md.first);
      unsigned int fromIndex;
      if(found == modsToIndex.end()) {
        fromIndex = modsToIndex.size();
        modsToIndex.emplace( md.first, fromIndex);
      } else {
        fromIndex = found->second;
      }
      for( auto const& dependsOn: md.second) {
        auto found = modsToIndex.find(dependsOn);
        unsigned int toIndex;
        if(found == modsToIndex.end()) {
          toIndex =modsToIndex.size();
          modsToIndex.emplace( dependsOn, toIndex);
        } else {
          toIndex = found->second;
        }
        edgeToPathMap[std::make_pair(fromIndex, toIndex)].push_back(kDataDependencyIndex);
      }
    }
    
    std::vector<std::string> pathNames;
    std::unordered_map<std::string, unsigned int> pathToIndexMap;
    for(auto const& path: iPaths) {
      unsigned int lastModuleIndex = kInvalidIndex;
      pathNames.push_back(path.first);
      unsigned int pathIndex = pathToIndexMap.size();
      pathToIndexMap.emplace(path.first, pathIndex);
      for( auto const& mod: path.second) {
        auto found = modsToIndex.find(mod);
        unsigned int index;
        if(found == modsToIndex.end()) {
          index =modsToIndex.size();
          modsToIndex.emplace( mod, index);
        } else {
          index = found->second;
        }
        if(lastModuleIndex != kInvalidIndex) {
          edgeToPathMap[std::make_pair(index, lastModuleIndex)].push_back(pathIndex);
        }
        lastModuleIndex = index;
      }
    }

    throwIfImproperDependencies(edgeToPathMap, pathNames, modsToIndex);
    
    return true;
    
  }
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(test_throwIfImproperDependencies);


void test_throwIfImproperDependencies::onePathNoCycleTest()
{
  {
    ModuleDependsOnMap md = { {"C", {"B"} },
                              {"B", {"A" } } };
    PathToModules paths = { {"p", {"A", "B", "C" } } };
    
    CPPUNIT_ASSERT( testCase(md,paths));
  }
  
  {
    ModuleDependsOnMap md = { };
    PathToModules paths = { {"p", {"A", "B", "C" } } };
    
    CPPUNIT_ASSERT( testCase(md,paths));
  }
  
  {
    //Circular dependency but not connected to any path
    ModuleDependsOnMap md = { {"E", {"F"} },
      {"F", {"G"} },
      {"G", {"E"} } };
    PathToModules paths = { {"p", {"A", "B", "C" } } };
    
    CPPUNIT_ASSERT( testCase(md,paths));
  }

}

void test_throwIfImproperDependencies::onePathHasCycleTest()
{
  {
    ModuleDependsOnMap md = { {"C", {"B"} },
      {"B", {"A" } } };
    {
      PathToModules paths = { {"p", {"B", "A", "C" } } };
    
      CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
    }
    {
      PathToModules paths = { {"p", {"B", "A" } } };
      
      CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
    }
    {
      PathToModules paths = { {"p", {"C", "A" } } };
      
      CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
    }
    {
      PathToModules paths = { {"p", {"C", "B" } } };
      
      CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
    }

  }
  
  {
    ModuleDependsOnMap md = { {"C", {"B"} } };
    {
      PathToModules paths = { {"p", {"C", "A", "B" } } };
      
      CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
    }
    {
      PathToModules paths = { {"p", {"A", "C", "B" } } };
      
      CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
    }
    {
      PathToModules paths = { {"p", {"C", "B", "A" } } };
      
      CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
    }
  }

}

void test_throwIfImproperDependencies::twoPathsNoCycleTest()
{
  {
    ModuleDependsOnMap md = { {"C", {"B"} } };

    {
      PathToModules paths = { {"p1", {"A", "B", "C" } },
                              {"p2", {"A", "B","C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }

    {
      PathToModules paths = { {"p1", {"A", "B", "C" } },
                              {"p2", {"B", "A","C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"A", "B", "C" } },
                              {"p2", {"B", "C","A"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"B", "A", "C" } },
                              {"p2", {"A", "B","C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"B", "A", "C" } },
                              {"p2", {"B", "A","C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"B", "A", "C" } },
                              {"p2", {"B", "C","A"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"B", "C", "A" } },
                              {"p2", {"A", "B","C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"B", "C", "A" } },
                              {"p2", {"B", "A","C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"B", "C", "A" } },
                              {"p2", {"B", "C","A"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
  }
  
  {
    //Test all possible 3 module combinations
    ModuleDependsOnMap md = {};
    std::vector<std::string> moduleName = {"A","B","C"};
    std::vector<std::string> pathModules;
    for(unsigned int i=0; i<3; ++i) {
      pathModules.push_back(moduleName[i]);
      for(unsigned int j=0; j<3; ++j) {
        if(j == i) {continue;}
        pathModules.push_back(moduleName[j]);
        for(unsigned int k=0; k<3; ++k) {
          if(j==k or i ==k) {continue;}
          pathModules.push_back(moduleName[k]);

          std::vector<std::string> path2Modules;
          for(unsigned int i=0; i<3; ++i) {
            path2Modules.push_back(moduleName[i]);
            for(unsigned int j=0; j<3; ++j) {
              if(j == i) {continue;}
              path2Modules.push_back(moduleName[j]);
              for(unsigned int k=0; k<3; ++k) {
                if(j==k or i ==k) {continue;}
                path2Modules.push_back(moduleName[k]);
                PathToModules paths;
                paths["p1"]=pathModules;
                paths["p2"]=path2Modules;
                CPPUNIT_ASSERT( testCase(md,paths));
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
    ModuleDependsOnMap md = { {"C", {"B"} },
                              {"B", {"A"}},
                              {"D", {"C"}} };
    
    {
      PathToModules paths = { {"p1", {"A", "C" } },
                              {"p2", {"B","D"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }

    {
      PathToModules paths = { {"p1", {"A", "D" } },
                              {"p2", {"B","C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"A", "B" } },
                              {"p2", {"C","D"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }

    {
      PathToModules paths = { {"p1", {"A"} },
                              {"p2", {"C","D"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"A"} },
                              {"p2", {"D"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"A"} },
                              {"p2", {"C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    
    {
      PathToModules paths = { {"p1", {"B" } },
                              {"p2", {"D"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }

    {
      PathToModules paths = { {"p1", {"B" } },
                              {"p2", {"C"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"C" } },
                              {"p2", {"D"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }

    {
      PathToModules paths = { {"p1", {"A", "C" } },
                              {"p2", {"A","D"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }
    {
      PathToModules paths = { {"p1", {"A", "C" } },
                              {"p2", {"A","D"} } };
      
      CPPUNIT_ASSERT( testCase(md,paths));
    }

  }
  
  {
    ModuleDependsOnMap md = { {"B", {"C"} },
      {"D", {"A" } } };
    PathToModules paths = { {"p1", {"A", "B" } },
      {"p2", {"C","D"}} };
    
    CPPUNIT_ASSERT( testCase(md,paths));
  }

  {
    ModuleDependsOnMap md = { {"B", {"E"} },
      {"E", {"C"}},
      {"D", {"F" }},
      {"F", {"A"}}};
    PathToModules paths = { {"p1", {"A", "B" } },
      {"p2", {"C","D"}} };
    
    CPPUNIT_ASSERT(testCase(md,paths));
  }

  {
    ModuleDependsOnMap md = { {"B", {"E"} },
      {"E", {"C"}},
      {"D", {"E" }},
      {"E", {"A"}}};
    PathToModules paths = { {"p1", {"A", "B"} },
                            {"p2", {"C","D"}} };
    
    CPPUNIT_ASSERT( testCase(md,paths));
  }

  {
    ModuleDependsOnMap md = { {"B", {"E"}},
                              {"C", {"E"}},
                              {"D", {"E"}},
                              {"A", {"E"}} };
    PathToModules paths = { {"p1", {"A", "B"} },
                            {"p2", {"C","D"}} };
    
    CPPUNIT_ASSERT( testCase(md,paths));
  }

  {
    //Simplified schedule which was failing
    ModuleDependsOnMap md = {{"H", {"A"}} };
    PathToModules paths = {
      {"reco", {"I", "A","H","G","F","E"} },
      {"val", {"E","D","C","B","A"}} };
  
  CPPUNIT_ASSERT( testCase(md,paths));
  }
  
}

void test_throwIfImproperDependencies::twoPathsWithCycleTest()
{
  
  {
    ModuleDependsOnMap md = { {"C", {"B"}},
                              {"A", {"D"}} };
    PathToModules paths = { {"p1", {"A","B"}},
                            {"p2", {"C","D"}} };
    
    CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
  }

  {
    //Add additional dependencies to test that they are ignored
    ModuleDependsOnMap md = {
      {"C", {"E","F","G","B"}},
      {"A", {"D"}} };
    PathToModules paths = {
      {"p1", {"A","B"}},
      {"p2", {"C","D"}} };
    
    CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = { {"C", {"B"}},
                              {"A", {"D"}} };
    PathToModules paths = { {"p1", {"A","B"}},
                            {"p2", {"C","D", "A"}} };
    
    CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = { {"C", {"B"}},
                              {"A", {"D"}},
                              {"B", {"A"}},
                              {"D", {"C"}} };
    PathToModules paths = { {"p1", {"A","B"}},
                            {"p2", {"C","D"}} };
    
    CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
  }

  {
    ModuleDependsOnMap md = { {"E", {"B"} },
                              {"C", {"E"}},
                              {"A", {"D"}}};
    PathToModules paths = { {"p1", {"A", "B" } },
                            {"p2", {"C","D"}} };
    
    CPPUNIT_ASSERT_THROW( testCase(md,paths), cms::Exception);
  }

}

