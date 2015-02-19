#include "FWCore/Framework/interface/PathsAndConsumesOfModules.h"

#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/src/Worker.h"
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
}
