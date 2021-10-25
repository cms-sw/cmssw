#include "FWCore/Framework/interface/PathsAndConsumesOfModules.h"

#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/ModuleProcessName.h"
#include "FWCore/Framework/interface/maker/Worker.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
namespace edm {

  PathsAndConsumesOfModules::PathsAndConsumesOfModules() = default;
  PathsAndConsumesOfModules::~PathsAndConsumesOfModules() = default;

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
    for (auto const& path : paths_) {
      schedule->moduleDescriptionsInPath(path, modulesOnPaths_.at(i), hint);
      if (!modulesOnPaths_.at(i).empty())
        ++hint;
      ++i;
    }

    modulesOnEndPaths_.resize(endPaths_.size());
    i = 0;
    hint = 0;
    for (auto const& endpath : endPaths_) {
      schedule->moduleDescriptionsInEndPath(endpath, modulesOnEndPaths_.at(i), hint);
      if (!modulesOnEndPaths_.at(i).empty())
        ++hint;
      ++i;
    }

    schedule->fillModuleAndConsumesInfo(allModuleDescriptions_,
                                        moduleIDToIndex_,
                                        modulesWhoseProductsAreConsumedBy_,
                                        modulesInPreviousProcessesWhoseProductsAreConsumedBy_,
                                        *preg);
  }

  void PathsAndConsumesOfModules::removeModules(std::vector<ModuleDescription const*> const& modules) {
    // First check that no modules on Paths are removed
    auto checkPath = [&modules](auto const& paths) {
      for (auto const& path : paths) {
        for (auto const& description : path) {
          if (std::find(modules.begin(), modules.end(), description) != modules.end()) {
            throw cms::Exception("Assert")
                << "PathsAndConsumesOfModules::removeModules() is trying to remove a module with label "
                << description->moduleLabel() << " id " << description->id() << " from a Path, this should not happen.";
          }
        }
      }
    };
    checkPath(modulesOnPaths_);
    checkPath(modulesOnEndPaths_);

    // Remove the modules and adjust the indices in idToIndex map
    for (auto iModule = 0U; iModule != allModuleDescriptions_.size(); ++iModule) {
      auto found = std::find(modules.begin(), modules.end(), allModuleDescriptions_[iModule]);
      if (found != modules.end()) {
        allModuleDescriptions_.erase(allModuleDescriptions_.begin() + iModule);
        for (auto iBranchType = 0U; iBranchType != NumBranchTypes; ++iBranchType) {
          modulesWhoseProductsAreConsumedBy_[iBranchType].erase(
              modulesWhoseProductsAreConsumedBy_[iBranchType].begin() + iModule);
        }
        modulesInPreviousProcessesWhoseProductsAreConsumedBy_.erase(
            modulesInPreviousProcessesWhoseProductsAreConsumedBy_.begin() + iModule);
        for (auto& idToIndex : moduleIDToIndex_) {
          if (idToIndex.second >= iModule) {
            idToIndex.second--;
          }
        }
        --iModule;
      }
    }
  }

  std::vector<ModuleProcessName> const& PathsAndConsumesOfModules::modulesInPreviousProcessesWhoseProductsAreConsumedBy(
      unsigned int moduleID) const {
    return modulesInPreviousProcessesWhoseProductsAreConsumedBy_.at(moduleIndex(moduleID));
  }

  ModuleDescription const* PathsAndConsumesOfModules::doModuleDescription(unsigned int moduleID) const {
    unsigned int dummy = 0;
    auto target = std::make_pair(moduleID, dummy);
    std::vector<std::pair<unsigned int, unsigned int>>::const_iterator iter =
        std::lower_bound(moduleIDToIndex_.begin(), moduleIDToIndex_.end(), target);
    if (iter == moduleIDToIndex_.end() || iter->first != moduleID) {
      throw Exception(errors::LogicError)
          << "PathsAndConsumesOfModules::moduleDescription: Unknown moduleID " << moduleID << "\n";
    }
    return allModuleDescriptions_.at(iter->second);
  }

  std::vector<ModuleDescription const*> const& PathsAndConsumesOfModules::doModulesOnPath(unsigned int pathIndex) const {
    return modulesOnPaths_.at(pathIndex);
  }

  std::vector<ModuleDescription const*> const& PathsAndConsumesOfModules::doModulesOnEndPath(
      unsigned int endPathIndex) const {
    return modulesOnEndPaths_.at(endPathIndex);
  }

  std::vector<ModuleDescription const*> const& PathsAndConsumesOfModules::doModulesWhoseProductsAreConsumedBy(
      unsigned int moduleID, BranchType branchType) const {
    return modulesWhoseProductsAreConsumedBy_[branchType].at(moduleIndex(moduleID));
  }

  std::vector<ConsumesInfo> PathsAndConsumesOfModules::doConsumesInfo(unsigned int moduleID) const {
    Worker const* worker = schedule_->allWorkers().at(moduleIndex(moduleID));
    return worker->consumesInfo();
  }

  unsigned int PathsAndConsumesOfModules::doLargestModuleID() const {
    // moduleIDToIndex_ is sorted, so last element has the largest ID
    return moduleIDToIndex_.empty() ? 0 : moduleIDToIndex_.back().first;
  }

  unsigned int PathsAndConsumesOfModules::moduleIndex(unsigned int moduleID) const {
    unsigned int dummy = 0;
    auto target = std::make_pair(moduleID, dummy);
    std::vector<std::pair<unsigned int, unsigned int>>::const_iterator iter =
        std::lower_bound(moduleIDToIndex_.begin(), moduleIDToIndex_.end(), target);
    if (iter == moduleIDToIndex_.end() || iter->first != moduleID) {
      throw Exception(errors::LogicError)
          << "PathsAndConsumesOfModules::moduleIndex: Unknown moduleID " << moduleID << "\n";
    }
    return iter->second;
  }
}  // namespace edm

namespace {
  // helper function for nonConsumedUnscheduledModules,
  void findAllConsumedModules(edm::PathsAndConsumesOfModulesBase const& iPnC,
                              edm::ModuleDescription const* module,
                              std::unordered_set<unsigned int>& consumedModules) {
    // If this node of the DAG has been processed already, no need to
    // reprocess again
    if (consumedModules.find(module->id()) != consumedModules.end()) {
      return;
    }
    consumedModules.insert(module->id());
    for (auto iBranchType = 0U; iBranchType != edm::NumBranchTypes; ++iBranchType) {
      for (auto const& c :
           iPnC.modulesWhoseProductsAreConsumedBy(module->id(), static_cast<edm::BranchType>(iBranchType))) {
        findAllConsumedModules(iPnC, c, consumedModules);
      }
    }
  }
}  // namespace

namespace edm {
  std::vector<ModuleDescription const*> nonConsumedUnscheduledModules(
      edm::PathsAndConsumesOfModulesBase const& iPnC, std::vector<ModuleProcessName>& consumedByChildren) {
    const std::string kTriggerResults("TriggerResults");

    std::vector<std::string> pathNames = iPnC.paths();
    const unsigned int kFirstEndPathIndex = pathNames.size();
    pathNames.insert(pathNames.end(), iPnC.endPaths().begin(), iPnC.endPaths().end());

    // The goal is to find modules that are not depended upon by
    // scheduled modules. To do that, we identify all modules that are
    // depended upon by scheduled modules, and do a set subtraction.
    //
    // First, denote all scheduled modules (i.e. in Paths and
    // EndPaths) as "consumers".
    std::vector<ModuleDescription const*> consumerModules;
    for (unsigned int pathIndex = 0; pathIndex != pathNames.size(); ++pathIndex) {
      std::vector<ModuleDescription const*> const* moduleDescriptions;
      if (pathIndex < kFirstEndPathIndex) {
        moduleDescriptions = &(iPnC.modulesOnPath(pathIndex));
      } else {
        moduleDescriptions = &(iPnC.modulesOnEndPath(pathIndex - kFirstEndPathIndex));
      }
      std::copy(moduleDescriptions->begin(), moduleDescriptions->end(), std::back_inserter(consumerModules));
    }

    // Then add TriggerResults, and all Paths and EndPaths themselves
    // to the set of "consumers" (even if they don't depend on any
    // data products, they must not be deleted). Also add anything
    // consumed by child SubProcesses to the set of "consumers".
    auto const& allModules = iPnC.allModules();
    for (auto const& description : allModules) {
      if (description->moduleLabel() == kTriggerResults or
          std::find(pathNames.begin(), pathNames.end(), description->moduleLabel()) != pathNames.end()) {
        consumerModules.push_back(description);
      } else if (std::binary_search(consumedByChildren.begin(),
                                    consumedByChildren.end(),
                                    ModuleProcessName{description->moduleLabel(), description->processName()}) or
                 std::binary_search(consumedByChildren.begin(),
                                    consumedByChildren.end(),
                                    ModuleProcessName{description->moduleLabel(), ""})) {
        consumerModules.push_back(description);
      }
    }

    // Find modules that have any data dependence path to any module
    // in consumerModules.
    std::unordered_set<unsigned int> consumedModules;
    for (auto& description : consumerModules) {
      findAllConsumedModules(iPnC, description, consumedModules);
    }

    // All other modules will then be classified as non-consumed, even
    // if they would have dependencies within them.
    std::vector<ModuleDescription const*> unusedModules;
    std::copy_if(allModules.begin(),
                 allModules.end(),
                 std::back_inserter(unusedModules),
                 [&consumedModules](ModuleDescription const* description) {
                   return consumedModules.find(description->id()) == consumedModules.end();
                 });
    return unusedModules;
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

  namespace {
    struct ModuleStatus {
      std::vector<unsigned int> dependsOn_;
      std::vector<unsigned int> pathsOn_;
      unsigned long long lastSearch = 0;
      bool onPath_ = false;
      bool wasRun_ = false;
    };

    struct PathStatus {
      std::vector<unsigned int> modulesOnPath_;
      unsigned long int activeModuleSlot_ = 0;
      unsigned long int nModules_ = 0;
      unsigned int index_ = 0;
      bool endPath_ = false;
    };

    class CircularDependencyException {};

    bool checkIfCanRun(unsigned long long searchIndex,
                       unsigned int iModuleToCheckID,
                       std::vector<ModuleStatus>& iModules,
                       std::vector<unsigned int>& stackTrace) {
      auto& status = iModules[iModuleToCheckID];
      if (status.wasRun_) {
        return true;
      }

      if (status.lastSearch == searchIndex) {
        //check to see if the module is already on the stack
        // checking searchIndex is insufficient as multiple modules
        // in this search may be dependent upon the same module
        auto itFound = std::find(stackTrace.begin(), stackTrace.end(), iModuleToCheckID);
        if (itFound != stackTrace.end()) {
          stackTrace.push_back(iModuleToCheckID);
          throw CircularDependencyException();
        }
        //we have already checked this module's dependencies during this search
        return false;
      }
      stackTrace.push_back(iModuleToCheckID);
      status.lastSearch = searchIndex;

      bool allDependenciesRan = true;
      for (auto index : status.dependsOn_) {
        auto& dep = iModules[index];
        if (dep.onPath_) {
          if (not dep.wasRun_) {
            allDependenciesRan = false;
          }
        } else if (not checkIfCanRun(searchIndex, index, iModules, stackTrace)) {
          allDependenciesRan = false;
        }
      }
      if (allDependenciesRan) {
        status.wasRun_ = true;
      }
      stackTrace.pop_back();

      return allDependenciesRan;
    }
  }  // namespace
  void checkForModuleDependencyCorrectness(edm::PathsAndConsumesOfModulesBase const& iPnC, bool iPrintDependencies) {
    constexpr auto kInvalidIndex = std::numeric_limits<unsigned int>::max();

    //Need to lookup ids to names quickly
    std::unordered_map<unsigned int, std::string> moduleIndexToNames;

    std::unordered_map<std::string, unsigned int> pathStatusInserterModuleLabelToModuleID;

    //for testing, state that TriggerResults is at the end of all paths
    const std::string kTriggerResults("TriggerResults");
    const std::string kPathStatusInserter("PathStatusInserter");
    const std::string kEndPathStatusInserter("EndPathStatusInserter");
    unsigned int kTriggerResultsIndex = kInvalidIndex;
    ModuleStatus triggerResultsStatus;
    unsigned int largestIndex = 0;
    for (auto const& description : iPnC.allModules()) {
      moduleIndexToNames.insert(std::make_pair(description->id(), description->moduleLabel()));
      if (kTriggerResults == description->moduleLabel()) {
        kTriggerResultsIndex = description->id();
      }
      if (description->id() > largestIndex) {
        largestIndex = description->id();
      }
      if (description->moduleName() == kPathStatusInserter) {
        triggerResultsStatus.dependsOn_.push_back(description->id());
      }
      if (description->moduleName() == kPathStatusInserter || description->moduleName() == kEndPathStatusInserter) {
        pathStatusInserterModuleLabelToModuleID[description->moduleLabel()] = description->id();
      }
    }

    std::vector<ModuleStatus> statusOfModules(largestIndex + 1);
    for (auto const& nameID : pathStatusInserterModuleLabelToModuleID) {
      statusOfModules[nameID.second].onPath_ = true;
    }
    if (kTriggerResultsIndex != kInvalidIndex) {
      statusOfModules[kTriggerResultsIndex] = std::move(triggerResultsStatus);
    }

    std::vector<PathStatus> statusOfPaths(iPnC.paths().size() + iPnC.endPaths().size());

    //If there are no paths, no modules will run so nothing to check
    if (statusOfPaths.empty()) {
      return;
    }

    {
      auto nPaths = iPnC.paths().size();
      for (unsigned int p = 0; p < nPaths; ++p) {
        auto& status = statusOfPaths[p];
        status.index_ = p;
        status.modulesOnPath_.reserve(iPnC.modulesOnPath(p).size() + 1);
        std::unordered_set<unsigned int> uniqueModules;
        for (auto const& mod : iPnC.modulesOnPath(p)) {
          if (uniqueModules.insert(mod->id()).second) {
            status.modulesOnPath_.push_back(mod->id());
            statusOfModules[mod->id()].onPath_ = true;
            statusOfModules[mod->id()].pathsOn_.push_back(p);
          }
        }
        status.nModules_ = uniqueModules.size() + 1;

        //add the PathStatusInserter at the end
        auto found = pathStatusInserterModuleLabelToModuleID.find(iPnC.paths()[p]);
        assert(found != pathStatusInserterModuleLabelToModuleID.end());
        status.modulesOnPath_.push_back(found->second);
      }
    }
    {
      auto offset = iPnC.paths().size();
      auto nPaths = iPnC.endPaths().size();
      for (unsigned int p = 0; p < nPaths; ++p) {
        auto& status = statusOfPaths[p + offset];
        status.endPath_ = true;
        status.index_ = p;
        status.modulesOnPath_.reserve(iPnC.modulesOnEndPath(p).size() + 1);
        std::unordered_set<unsigned int> uniqueModules;
        for (auto const& mod : iPnC.modulesOnEndPath(p)) {
          if (uniqueModules.insert(mod->id()).second) {
            status.modulesOnPath_.push_back(mod->id());
            statusOfModules[mod->id()].onPath_ = true;
            statusOfModules[mod->id()].pathsOn_.push_back(p + offset);
          }
        }
        status.nModules_ = uniqueModules.size() + 1;

        //add the EndPathStatusInserter at the end
        auto found = pathStatusInserterModuleLabelToModuleID.find(iPnC.endPaths()[p]);
        assert(found != pathStatusInserterModuleLabelToModuleID.end());
        status.modulesOnPath_.push_back(found->second);
      }
    }

    for (auto const& description : iPnC.allModules()) {
      unsigned int const moduleIndex = description->id();
      auto const& dependentModules = iPnC.modulesWhoseProductsAreConsumedBy(moduleIndex);
      auto& deps = statusOfModules[moduleIndex];
      deps.dependsOn_.reserve(dependentModules.size());
      for (auto const& depDescription : dependentModules) {
        if (iPrintDependencies) {
          edm::LogAbsolute("ModuleDependency")
              << "ModuleDependency '" << description->moduleLabel() << "' depends on data products from module '"
              << depDescription->moduleLabel() << "'";
        }
        deps.dependsOn_.push_back(depDescription->id());
      }
    }

    unsigned int nPathsFinished = 0;

    //if a circular dependency exception happens, stackTrace has the info
    std::vector<unsigned int> stackTrace;
    bool madeForwardProgress = true;
    try {
      //'simulate' the running of the paths. On each step mark each module as 'run'
      // if all the module's dependencies were fulfilled in a previous step
      unsigned long long searchIndex = 0;
      while (madeForwardProgress and nPathsFinished != statusOfPaths.size()) {
        madeForwardProgress = false;
        for (auto& p : statusOfPaths) {
          //the path has already completed in an earlier pass
          if (p.activeModuleSlot_ == p.nModules_) {
            continue;
          }
          ++searchIndex;
          bool didRun = checkIfCanRun(searchIndex, p.modulesOnPath_[p.activeModuleSlot_], statusOfModules, stackTrace);
          if (didRun) {
            madeForwardProgress = true;
            ++p.activeModuleSlot_;
            if (p.activeModuleSlot_ == p.nModules_) {
              ++nPathsFinished;
            }
          }
        }
      }
    } catch (CircularDependencyException const&) {
      //the last element in stackTrace must appear somewhere earlier in stackTrace
      std::ostringstream oStr;

      unsigned int lastIndex = stackTrace.front();
      bool firstSkipped = false;
      for (auto id : stackTrace) {
        if (firstSkipped) {
          oStr << "  module '" << moduleIndexToNames[lastIndex] << "' depends on " << moduleIndexToNames[id] << "\n";
        } else {
          firstSkipped = true;
        }
        lastIndex = id;
      }
      throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
          << "Circular module dependency found in configuration\n"
          << oStr.str();
    }

    auto pathName = [&](PathStatus const& iP) {
      if (iP.endPath_) {
        return iPnC.endPaths()[iP.index_];
      }
      return iPnC.paths()[iP.index_];
    };

    //The program would deadlock
    if (not madeForwardProgress) {
      std::ostringstream oStr;
      auto modIndex = std::numeric_limits<unsigned int>::max();
      unsigned int presentPath;
      for (auto itP = statusOfPaths.begin(); itP != statusOfPaths.end(); ++itP) {
        auto const& p = *itP;
        if (p.activeModuleSlot_ == p.nModules_) {
          continue;
        }
        //this path is stuck
        modIndex = p.modulesOnPath_[p.activeModuleSlot_];
        presentPath = itP - statusOfPaths.begin();
        break;
      }
      //NOTE the following should always be true as at least 1 path should be stuc.
      // I've added the condition just to be paranoid.
      if (modIndex != std::numeric_limits<unsigned int>::max()) {
        struct ProgressInfo {
          ProgressInfo(unsigned int iMod, unsigned int iPath, bool iPreceeds = false)
              : moduleIndex_(iMod), pathIndex_(iPath), preceeds_(iPreceeds) {}

          ProgressInfo(unsigned int iMod) : moduleIndex_(iMod), pathIndex_{}, preceeds_(false) {}

          unsigned int moduleIndex_ = std::numeric_limits<unsigned int>::max();
          std::optional<unsigned int> pathIndex_;
          bool preceeds_;

          bool operator==(ProgressInfo const& iOther) const {
            return moduleIndex_ == iOther.moduleIndex_ and pathIndex_ == iOther.pathIndex_;
          }
        };

        std::vector<ProgressInfo> progressTrace;
        progressTrace.emplace_back(modIndex, presentPath);

        //The following starts from the first found unrun module on a path. It then finds
        // the first modules it depends on that was not run. If that module is on a Task
        // it then repeats the check for that module's dependencies. If that module is on
        // a path, it checks to see if that module is the first unrun module of a path
        // and if so it repeats the check for that module's dependencies, if not it
        // checks the dependencies of the stuck module on that path.
        // Eventually, all these checks should allow us to find a cycle of modules.

        //NOTE: the only way foundUnrunModule should ever by false by the end of the
        // do{}while loop is if there is a bug in the algorithm. I've included it to
        // try to avoid that case causing an infinite loop in the program.
        bool foundUnrunModule;
        do {
          //check dependencies looking for stuff not run and on a path
          foundUnrunModule = false;
          for (auto depMod : statusOfModules[modIndex].dependsOn_) {
            auto const& depStatus = statusOfModules[depMod];
            if (not depStatus.wasRun_ and depStatus.onPath_) {
              foundUnrunModule = true;
              //last run on a path?
              bool lastOnPath = false;
              unsigned int foundPath;
              for (auto pathOn : depStatus.pathsOn_) {
                auto const& depPaths = statusOfPaths[pathOn];
                if (depPaths.modulesOnPath_[depPaths.activeModuleSlot_] == depMod) {
                  lastOnPath = true;
                  foundPath = pathOn;
                  break;
                }
              }
              if (lastOnPath) {
                modIndex = depMod;
                progressTrace.emplace_back(modIndex, foundPath);
              } else {
                //some earlier module on the same path is stuck
                progressTrace.emplace_back(depMod, depStatus.pathsOn_[0]);
                auto const& depPath = statusOfPaths[depStatus.pathsOn_[0]];
                modIndex = depPath.modulesOnPath_[depPath.activeModuleSlot_];
                progressTrace.emplace_back(modIndex, depStatus.pathsOn_[0], true);
              }
              break;
            }
          }
          if (not foundUnrunModule) {
            //check unscheduled modules
            for (auto depMod : statusOfModules[modIndex].dependsOn_) {
              auto const& depStatus = statusOfModules[depMod];
              if (not depStatus.wasRun_ and not depStatus.onPath_) {
                foundUnrunModule = true;
                progressTrace.emplace_back(depMod);
                modIndex = depMod;
                break;
              }
            }
          }
        } while (foundUnrunModule and (0 == std::count(progressTrace.begin(),
                                                       progressTrace.begin() + progressTrace.size() - 1,
                                                       progressTrace.back())));

        auto printTrace = [&](auto& oStr, auto itBegin, auto itEnd) {
          for (auto itTrace = itBegin; itTrace != itEnd; ++itTrace) {
            if (itTrace != itBegin) {
              if (itTrace->preceeds_) {
                oStr << " and follows module '" << moduleIndexToNames[itTrace->moduleIndex_] << "' on the path\n";
              } else {
                oStr << " and depends on module '" << moduleIndexToNames[itTrace->moduleIndex_] << "'\n";
              }
            }
            if (itTrace + 1 != itEnd) {
              if (itTrace->pathIndex_) {
                oStr << "  module '" << moduleIndexToNames[itTrace->moduleIndex_] << "' is on path '"
                     << pathName(statusOfPaths[*itTrace->pathIndex_]) << "'";
              } else {
                oStr << "  module '" << moduleIndexToNames[itTrace->moduleIndex_] << "' is in a task";
              }
            }
          }
        };

        if (not foundUnrunModule) {
          //If we get here, this suggests a problem with either the algorithm that finds problems or the algorithm
          // that attempts to report the problem
          oStr << "Algorithm Error, unable to find problem. Contact framework group.\n Traced problem this far\n";
          printTrace(oStr, progressTrace.begin(), progressTrace.end());
        } else {
          printTrace(
              oStr, std::find(progressTrace.begin(), progressTrace.end(), progressTrace.back()), progressTrace.end());
        }
      }
      //the schedule deadlocked
      throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
          << "The Path/EndPath configuration could cause the job to deadlock\n"
          << oStr.str();
    }

    //NOTE: although the following conditions are not needed for safe running, they are
    // policy choices the collaboration has made.

    //Check to see if for each path if the order of the modules is correct based on dependencies
    for (auto& p : statusOfPaths) {
      for (unsigned long int i = 0; p.nModules_ > 0 and i < p.nModules_ - 1; ++i) {
        auto moduleID = p.modulesOnPath_[i];
        if (not statusOfModules[moduleID].dependsOn_.empty()) {
          for (unsigned long int j = i + 1; j < p.nModules_; ++j) {
            auto testModuleID = p.modulesOnPath_[j];
            for (auto depModuleID : statusOfModules[moduleID].dependsOn_) {
              if (depModuleID == testModuleID) {
                throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
                    << "Dependent module later on Path\n"
                    << "  module '" << moduleIndexToNames[moduleID] << "' depends on '"
                    << moduleIndexToNames[depModuleID] << "' which is later on path " << pathName(p);
              }
            }
          }
        }
      }
    }

    //HLT wants all paths to be equivalent. If a path has a module A that needs data from module B and module B appears on one path
    // as module A then B must appear on ALL paths that have A.
    unsigned int modIndex = 0;
    for (auto& mod : statusOfModules) {
      for (auto& depIndex : mod.dependsOn_) {
        std::size_t count = 0;
        std::size_t nonEndPaths = 0;
        for (auto modPathID : mod.pathsOn_) {
          if (statusOfPaths[modPathID].endPath_) {
            continue;
          }
          ++nonEndPaths;
          for (auto depPathID : statusOfModules[depIndex].pathsOn_) {
            if (depPathID == modPathID) {
              ++count;
              break;
            }
          }
        }
        if (count != 0 and count != nonEndPaths) {
          std::ostringstream onStr;
          std::ostringstream missingStr;

          for (auto modPathID : mod.pathsOn_) {
            if (statusOfPaths[modPathID].endPath_) {
              continue;
            }
            bool found = false;
            for (auto depPathID : statusOfModules[depIndex].pathsOn_) {
              if (depPathID == modPathID) {
                found = true;
              }
            }
            auto& s = statusOfPaths[modPathID];
            if (found) {
              onStr << pathName(s) << " ";
            } else {
              missingStr << pathName(s) << " ";
            }
          }
          throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
              << "Paths are non consistent\n"
              << "  module '" << moduleIndexToNames[modIndex] << "' depends on '" << moduleIndexToNames[depIndex]
              << "' which appears on paths\n  " << onStr.str() << "\nbut is missing from\n  " << missingStr.str();
        }
      }
      ++modIndex;
    }
  }
}  // namespace edm
