#include "FWCore/Framework/interface/PathsAndConsumesOfModules.h"

#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/src/Worker.h"
#include "throwIfImproperDependencies.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
namespace edm {

  PathsAndConsumesOfModules::~PathsAndConsumesOfModules() {
  }

  void PathsAndConsumesOfModules::initialize(Schedule const* schedule, std::shared_ptr<ProductRegistry const> preg) {

    schedule_ = schedule;
    preg_ = preg;

    paths_.clear();
    schedule->triggerPaths(paths_);

    endPaths_.clear();
    schedule->endPaths(endPaths_);

    modulesOnPaths_.resize(paths_.size());
    unsigned int i = 0;
    unsigned int hint = 0;
    for(auto const& path : paths_) {
      schedule->moduleDescriptionsInPath(path, modulesOnPaths_.at(i), hint);
      if(!modulesOnPaths_.at(i).empty()) ++hint;
      ++i;
    }

    modulesOnEndPaths_.resize(endPaths_.size());
    i = 0;
    hint = 0;
    for(auto const& endpath : endPaths_) {
      schedule->moduleDescriptionsInEndPath(endpath, modulesOnEndPaths_.at(i), hint);
      if(!modulesOnEndPaths_.at(i).empty()) ++hint;
      ++i;
    }

    schedule->fillModuleAndConsumesInfo(allModuleDescriptions_,
                                        moduleIDToIndex_,
                                        modulesWhoseProductsAreConsumedBy_,
                                        *preg);
  }

  ModuleDescription const*
  PathsAndConsumesOfModules::doModuleDescription(unsigned int moduleID) const {
    unsigned int dummy = 0;
    auto target = std::make_pair(moduleID, dummy);
    std::vector<std::pair<unsigned int, unsigned int> >::const_iterator iter =
      std::lower_bound(moduleIDToIndex_.begin(), moduleIDToIndex_.end(), target);
    if (iter == moduleIDToIndex_.end() || iter->first != moduleID) {
      throw Exception(errors::LogicError)
        << "PathsAndConsumesOfModules::moduleDescription: Unknown moduleID\n";
    }
    return allModuleDescriptions_.at(iter->second);
  }

  std::vector<ModuleDescription const*> const&
  PathsAndConsumesOfModules::doModulesOnPath(unsigned int pathIndex) const {
    return modulesOnPaths_.at(pathIndex);
  }

  std::vector<ModuleDescription const*> const&
  PathsAndConsumesOfModules::doModulesOnEndPath(unsigned int endPathIndex) const {
    return modulesOnEndPaths_.at(endPathIndex);
  }

  std::vector<ModuleDescription const*> const&
  PathsAndConsumesOfModules::doModulesWhoseProductsAreConsumedBy(unsigned int moduleID) const {
    return modulesWhoseProductsAreConsumedBy_.at(moduleIndex(moduleID));
  }

  std::vector<ConsumesInfo> 
  PathsAndConsumesOfModules::doConsumesInfo(unsigned int moduleID) const {
    Worker const* worker = schedule_->allWorkers().at(moduleIndex(moduleID));
    return worker->consumesInfo();
  }

  unsigned int
  PathsAndConsumesOfModules::moduleIndex(unsigned int moduleID) const {
    unsigned int dummy = 0;
    auto target = std::make_pair(moduleID, dummy);
    std::vector<std::pair<unsigned int, unsigned int> >::const_iterator iter =
      std::lower_bound(moduleIDToIndex_.begin(), moduleIDToIndex_.end(), target);
    if (iter == moduleIDToIndex_.end() || iter->first != moduleID) {
      throw Exception(errors::LogicError)
        << "PathsAndConsumesOfModules::moduleIndex: Unknown moduleID\n";
    }
    return iter->second;
  }
  
  //====================================
  // checkForCorrectness algorithm
  //
  // The code creates a 'dependency' graph between all
  // modules. A module depends on another module if
  // 1) it 'consumes' data produced by that module
  // 2) it appears directly after the module within a Path
  //
  // If there is a cycle in the 'dependency' graph then
  // the schedule may be unrunnable. The schedule is still
  // runnable if all cycles have at least two edges which
  // connect modules only by Path dependencies (i.e. not
  // linked by a data dependency).
  //
  //  Example 1:
  //  C consumes data from B
  //  Path 1: A + B + C
  //  Path 2: B + C + A
  //
  //  Cycle: A after C [p2], C consumes B, B after A [p1]
  //  Since this cycle has 2 path only edges it is OK since
  //  A and (B+C) are independent so their run order doesn't matter
  //
  //  Example 2:
  //  B consumes A
  //  C consumes B
  //  Path: C + A
  //
  //  Cycle: A after C [p], C consumes B, B consumes A
  //  Since this cycle has 1 path only edge it is unrunnable.
  //
  //  Example 3:
  //  A consumes B
  //  B consumes C
  //  C consumes A
  //  (no Path since unscheduled execution)
  //
  //  Cycle: A consumes B, B consumes C, C consumes A
  //  Since this cycle has 0 path only edges it is unrunnable.
  //====================================
  

  void checkForModuleDependencyCorrectness(edm::PathsAndConsumesOfModulesBase const& iPnC,
                                           bool iPrintDependencies) {
    using namespace edm::graph;
    //Need to lookup ids to names quickly
    std::unordered_map<unsigned int, std::string> moduleIndexToNames;
    
    //for testing, state that TriggerResults is at the end of all paths
    const std::string kTriggerResults("TriggerResults");
    unsigned int kTriggerResultsIndex = kInvalidIndex;
    unsigned int largestIndex = 0;
    unsigned int kPathToTriggerResultsDependencyLastIndex = kInvalidIndex;
    for(auto const& description: iPnC.allModules()) {
      moduleIndexToNames.insert(std::make_pair(description->id(),
                                               description->moduleLabel()));
      if(kTriggerResults == description->moduleLabel()) {
        kTriggerResultsIndex = description->id();
      }
      if(description->id() > largestIndex) {
        largestIndex = description->id();
      }
    }
    kPathToTriggerResultsDependencyLastIndex = largestIndex ;


    /*
    {
      //We need to explicitly check that modules on Paths do not try to read data from
      // Modules which are only on EndPaths. The circular dependency finder has been
      // known to miss these.
      std::unordered_set<unsigned int> modulesOnlyOnEndPaths;
      auto const& endPaths = iPnC.endPaths();
      for( unsigned int pathIndex = 0; pathIndex != endPaths.size(); ++pathIndex) {
        auto const& moduleDescriptions = iPnC.modulesOnEndPath(pathIndex);
        for(auto const& description: moduleDescriptions) {
          modulesOnlyOnEndPaths.insert(description->id());
        }
      }

      std::unordered_set<unsigned int> modulesOnPaths;
      auto const& paths = iPnC.paths();
      for( unsigned int pathIndex = 0; pathIndex != paths.size(); ++pathIndex) {
        auto const& moduleDescriptions = iPnC.modulesOnPath(pathIndex);
        for(auto const& description: moduleDescriptions) {
          auto itFind =modulesOnlyOnEndPaths.find(description->id());
          if(modulesOnlyOnEndPaths.end() != itFind) {
            modulesOnlyOnEndPaths.erase(itFind);
          }
          modulesOnPaths.insert(description->id());
        }
      }
      
      for(auto moduleIndex : modulesOnPaths) {
        auto const& dependentModules = iPnC.modulesWhoseProductsAreConsumedBy(moduleIndex);
        for(auto const& depDescription: dependentModules) {
          auto itFind = modulesOnlyOnEndPaths.find(depDescription->id());
          if(itFind != modulesOnlyOnEndPaths.end()) {
            throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n") 
              <<"The module "<<moduleIndexToNames[moduleIndex]<<" is on a Path and depends on data from module "
              <<moduleIndexToNames[depDescription->id()]<<" which is on an EndPath.";
          }
        }
      }

    }
     */

    //If a module to module dependency comes from a path, remember which path
    EdgeToPathMap edgeToPathMap;

    //Need to be able to quickly look up which paths a module appears on
    std::unordered_map<unsigned int, std::vector<unsigned int>> moduleIndexToPathIndex;

    //determine the path dependencies
    std::vector<std::string> pathNames = iPnC.paths();
    const unsigned int kFirstEndPathIndex = pathNames.size();

    const std::string kPathEnded("@PathEnded");
    const std::string kEndPathStart("@EndPathStart");
    
    //The finished processing depends on all paths and end paths
    const std::string kFinishedProcessing("@FinishedProcessing");
    const unsigned int kFinishedProcessingIndex{0};
    moduleIndexToNames.insert(std::make_pair(kFinishedProcessingIndex,
                                             kFinishedProcessing));


    pathNames.insert(pathNames.end(), iPnC.endPaths().begin(), iPnC.endPaths().end());
    std::vector<std::vector<unsigned int>> pathIndexToModuleIndexOrder(pathNames.size());
    {
      
      for(unsigned int pathIndex = 0; pathIndex != pathNames.size(); ++pathIndex) {
        std::set<unsigned int> alreadySeenIndex;
        
        std::vector<ModuleDescription const*> const* moduleDescriptions;
        if(pathIndex < kFirstEndPathIndex) {
          moduleDescriptions= &(iPnC.modulesOnPath(pathIndex));
        } else {
          moduleDescriptions= &(iPnC.modulesOnEndPath(pathIndex-kFirstEndPathIndex));
        }
        unsigned int lastModuleIndex = kInvalidIndex;
        auto& pathOrder =pathIndexToModuleIndexOrder[pathIndex];
        pathOrder.reserve(moduleDescriptions->size() + 1);
        for(auto const& description: *moduleDescriptions) {
          auto found = alreadySeenIndex.insert(description->id());
          if(found.second) {
            //first time for this path
            unsigned int const moduleIndex = description->id();
            pathOrder.push_back(moduleIndex);
            auto& paths =moduleIndexToPathIndex[moduleIndex];
            paths.push_back(pathIndex);
            if(lastModuleIndex  != kInvalidIndex ) {
              edgeToPathMap[std::make_pair(moduleIndex,lastModuleIndex)].push_back(pathIndex);
            }
            lastModuleIndex = moduleIndex;
          }
        }
        //Have TriggerResults depend on the end of all paths 
        // Have all EndPaths depend on TriggerResults
        if( pathIndex < kFirstEndPathIndex) {
          if( (lastModuleIndex  != kInvalidIndex) ) {
            ++kPathToTriggerResultsDependencyLastIndex;
            edgeToPathMap[std::make_pair(kPathToTriggerResultsDependencyLastIndex,lastModuleIndex)].push_back(pathIndex);
            moduleIndexToNames.insert(std::make_pair(kPathToTriggerResultsDependencyLastIndex,
                                                     kPathEnded));
            if (kTriggerResultsIndex != kInvalidIndex) {
              edgeToPathMap[std::make_pair(kTriggerResultsIndex, kPathToTriggerResultsDependencyLastIndex)].push_back(kDataDependencyIndex);
            }
            //Need to make dependency for finished process
            edgeToPathMap[std::make_pair(kFinishedProcessingIndex, kPathToTriggerResultsDependencyLastIndex)].push_back(kDataDependencyIndex);
            pathOrder.push_back(kPathToTriggerResultsDependencyLastIndex);
          }
        } else {
          if( (not moduleDescriptions->empty()) ) {
            if (kTriggerResultsIndex != kInvalidIndex) {
              ++kPathToTriggerResultsDependencyLastIndex;
              edgeToPathMap[std::make_pair(moduleDescriptions->front()->id(),kPathToTriggerResultsDependencyLastIndex)].push_back(pathIndex);
              moduleIndexToNames.insert(std::make_pair(kPathToTriggerResultsDependencyLastIndex,
                                                       kEndPathStart));
              edgeToPathMap[std::make_pair(kPathToTriggerResultsDependencyLastIndex,kTriggerResultsIndex)].push_back(kDataDependencyIndex);
              pathOrder.insert(pathOrder.begin(),kPathToTriggerResultsDependencyLastIndex);
            }
            //Need to make dependency for finished process
            ++kPathToTriggerResultsDependencyLastIndex;
            edgeToPathMap[std::make_pair(kPathToTriggerResultsDependencyLastIndex,lastModuleIndex)].push_back(pathIndex);
            moduleIndexToNames.insert(std::make_pair(kPathToTriggerResultsDependencyLastIndex,
                                                     kPathEnded));
            edgeToPathMap[std::make_pair(kFinishedProcessingIndex, kPathToTriggerResultsDependencyLastIndex)].push_back(kDataDependencyIndex);
            pathOrder.push_back(kPathToTriggerResultsDependencyLastIndex);
          }
        }
      }
    }
    {

      //determine the data dependencies
      for(auto const& description: iPnC.allModules()) {
        unsigned int const moduleIndex = description->id();
        auto const& dependentModules = iPnC.modulesWhoseProductsAreConsumedBy(moduleIndex);
        for(auto const& depDescription: dependentModules) {
          if(iPrintDependencies) {
            edm::LogAbsolute("ModuleDependency") << "ModuleDependency '" << description->moduleLabel() <<
            "' depends on data products from module '" << depDescription->moduleLabel()<<"'";
          }
          //see if all paths containing this module also contain the dependent module earlier in the path
          // if it does, then treat this only as a path dependency and not a data dependency as this
          // simplifies the circular dependency checking logic
          auto depID =depDescription->id();
          auto itPathsFound = moduleIndexToPathIndex.find(moduleIndex);
          bool keepDataDependency = true;
          auto itDepsPathsFound =moduleIndexToPathIndex.find(depID);
          if(itPathsFound != moduleIndexToPathIndex.end() and itDepsPathsFound != moduleIndexToPathIndex.end()) {
            keepDataDependency = false;
            for(auto const pathIndex: itPathsFound->second) {
              for(auto idToCheck: pathIndexToModuleIndexOrder[pathIndex]) {
                if(idToCheck == depID) {
                  //found dependent module first so check next path
                  break;
                }
                if(idToCheck == moduleIndex) {
                  //did not find dependent module earlier on path so
                  // must keep data dependency
                  keepDataDependency = true;
                  break;
                }
              }
              if(keepDataDependency) {
                break;
              }
            }
          }
          if(keepDataDependency) {
            edgeToPathMap[std::make_pair(moduleIndex, depID)].push_back(kDataDependencyIndex);
          }
        }
      }
    }
    graph::throwIfImproperDependencies(edgeToPathMap,pathIndexToModuleIndexOrder,pathNames,moduleIndexToNames);
  }
}
