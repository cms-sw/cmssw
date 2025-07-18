#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"

#include "FWCore/ServiceRegistry/interface/ESModuleConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesESInfo.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesInfo.h"

namespace edm {

  PathsAndConsumesOfModulesBase::~PathsAndConsumesOfModulesBase() {}

  std::vector<ModuleConsumesInfo> PathsAndConsumesOfModulesBase::moduleConsumesInfos(unsigned int moduleID) const {
    return doModuleConsumesInfos(moduleID);
  }

  std::vector<ModuleConsumesESInfo> PathsAndConsumesOfModulesBase::moduleConsumesESInfos(unsigned int moduleID) const {
    return doModuleConsumesESInfos(moduleID);
  }

  std::vector<std::vector<ESModuleConsumesInfo>> PathsAndConsumesOfModulesBase::esModuleConsumesInfos(
      unsigned int esModuleID) const {
    return doESModuleConsumesInfos(esModuleID);
  }

}  // namespace edm
