#include "FWCore/Framework/interface/ScheduleItems.h"

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/SubProcessParentageHelper.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "FWCore/AbstractServices/interface/ResourceInformation.h"
#include "FWCore/Common/interface/SubProcessBlockHelper.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/src/CommonParams.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/SignallingProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include <memory>

#include <set>
#include <string>
#include <vector>

namespace edm {
  ScheduleItems::ScheduleItems()
      : actReg_(std::make_shared<ActivityRegistry>()),
        preg_(std::make_shared<SignallingProductRegistry>()),
        branchIDListHelper_(std::make_shared<BranchIDListHelper>()),
        thinnedAssociationsHelper_(std::make_shared<ThinnedAssociationsHelper>()),
        subProcessParentageHelper_(),
        act_table_(),
        processConfiguration_() {}

  ScheduleItems::ScheduleItems(ProductRegistry const& preg,
                               SubProcess const& om,
                               SubProcessBlockHelper& subProcessBlockHelper,
                               ProcessBlockHelperBase const& parentProcessBlockHelper)
      : actReg_(std::make_shared<ActivityRegistry>()),
        preg_(std::make_shared<SignallingProductRegistry>(preg)),
        branchIDListHelper_(std::make_shared<BranchIDListHelper>()),
        thinnedAssociationsHelper_(std::make_shared<ThinnedAssociationsHelper>()),
        subProcessParentageHelper_(std::make_shared<SubProcessParentageHelper>()),
        act_table_(),
        processConfiguration_() {
    for (auto& item : preg_->productListUpdator()) {
      ProductDescription& prod = item.second;
      prod.setOnDemand(false);
      prod.setProduced(false);
    }

    // Mark dropped branches as dropped in the product registry.
    std::set<BranchID> keptBranches;
    SelectedProducts const& keptVectorP = om.keptProducts()[InProcess];
    for (auto const& item : keptVectorP) {
      ProductDescription const& desc = *item.first;
      keptBranches.insert(desc.branchID());
    }
    SelectedProducts const& keptVectorR = om.keptProducts()[InRun];
    for (auto const& item : keptVectorR) {
      ProductDescription const& desc = *item.first;
      keptBranches.insert(desc.branchID());
    }
    SelectedProducts const& keptVectorL = om.keptProducts()[InLumi];
    for (auto const& item : keptVectorL) {
      ProductDescription const& desc = *item.first;
      keptBranches.insert(desc.branchID());
    }
    SelectedProducts const& keptVectorE = om.keptProducts()[InEvent];
    for (auto const& item : keptVectorE) {
      ProductDescription const& desc = *item.first;
      keptBranches.insert(desc.branchID());
    }
    for (auto& item : preg_->productListUpdator()) {
      ProductDescription& prod = item.second;
      if (keptBranches.find(prod.branchID()) == keptBranches.end()) {
        prod.setDropped(true);
      }
    }
    subProcessBlockHelper.updateFromParentProcess(parentProcessBlockHelper, preg_->registry());
  }

  ServiceToken ScheduleItems::initServices(std::vector<ParameterSet>& pServiceSets,
                                           ParameterSet& parameterSet,
                                           ServiceToken const& iToken,
                                           serviceregistry::ServiceLegacy iLegacy,
                                           bool associate) {
    //create the services
    ServiceToken token(ServiceRegistry::createSet(pServiceSets, iToken, iLegacy, associate));

    //see if any of the Services have to have their PSets stored
    for (auto const& item : pServiceSets) {
      if (item.exists("@save_config")) {
        parameterSet.addParameter(item.getParameter<std::string>("@service_type"), item);
      }
    }
    // Copy slots that hold all the registered callback functions like
    // PostBeginJob into an ActivityRegistry
    token.copySlotsTo(*actReg_);
    return token;
  }

  ServiceToken ScheduleItems::addTNS(ParameterSet const& parameterSet, ServiceToken const& token) {
    // This is ugly: pull out the trigger path pset and
    // create a service and extra token for it

    typedef service::TriggerNamesService TNS;
    typedef serviceregistry::ServiceWrapper<TNS> w_TNS;

    auto tnsptr = std::make_shared<w_TNS>(std::make_unique<TNS>(parameterSet));

    return ServiceRegistry::createContaining(tnsptr, token, serviceregistry::kOverlapIsError);
  }

  std::shared_ptr<CommonParams> ScheduleItems::initMisc(ParameterSet& parameterSet) {
    edm::Service<edm::ResourceInformation> resourceInformationService;
    edm::HardwareResourcesDescription hwResources;
    if (resourceInformationService.isAvailable()) {
      auto const& selectedAccelerators =
          parameterSet.getUntrackedParameter<std::vector<std::string>>("@selected_accelerators");
      resourceInformationService->setSelectedAccelerators(selectedAccelerators);
      // HardwareResourcesDescription is optional here in order to not
      // require ResourceInformationService in TestProcessor
      hwResources = resourceInformationService->hardwareResourcesDescription();
    }

    act_table_ = std::make_unique<ExceptionToActionTable>(parameterSet);
    std::string processName = parameterSet.getParameter<std::string>("@process_name");
    std::string releaseVersion;
    if (parameterSet.existsAs<std::string>("@special_override_release_version_only_for_testing", false)) {
      releaseVersion =
          parameterSet.getUntrackedParameter<std::string>("@special_override_release_version_only_for_testing");
    } else {
      releaseVersion = getReleaseVersion();
    }
    // propagate_const<T> has no reset() function
    processConfiguration_ = std::make_shared<ProcessConfiguration>(processName, releaseVersion, hwResources);
    auto common = std::make_shared<CommonParams>(
        parameterSet.getUntrackedParameterSet("maxEvents").getUntrackedParameter<int>("input"),
        parameterSet.getUntrackedParameterSet("maxLuminosityBlocks").getUntrackedParameter<int>("input"),
        parameterSet.getUntrackedParameterSet("maxSecondsUntilRampdown").getUntrackedParameter<int>("input"));
    return common;
  }

  std::unique_ptr<Schedule> ScheduleItems::initSchedule(ParameterSet& parameterSet,
                                                        bool hasSubprocesses,
                                                        PreallocationConfiguration const& config,
                                                        ProcessContext const* processContext,
                                                        ModuleTypeResolverMaker const* typeResolverMaker,
                                                        ProcessBlockHelperBase& processBlockHelper) {
    auto& tns = ServiceRegistry::instance().get<service::TriggerNamesService>();
    auto ret = std::make_unique<Schedule>(parameterSet,
                                          tns,
                                          *preg_,
                                          *act_table_,
                                          actReg_,
                                          processConfiguration(),
                                          config,
                                          processContext,
                                          typeResolverMaker);
    ret->finishSetup(parameterSet,
                     tns,
                     *preg_,
                     *branchIDListHelper_,
                     processBlockHelper,
                     *thinnedAssociationsHelper_,
                     subProcessParentageHelper_ ? subProcessParentageHelper_.get() : nullptr,
                     actReg_,
                     processConfiguration(),
                     hasSubprocesses,
                     config,
                     processContext);
    return ret;
  }

  ScheduleItems::MadeModules ScheduleItems::initModules(ParameterSet& parameterSet,
                                                        service::TriggerNamesService const& tns,
                                                        PreallocationConfiguration const& config,
                                                        ProcessContext const* processContext,
                                                        ModuleTypeResolverMaker const* typeResolverMaker) {
    return MadeModules(std::make_unique<Schedule>(parameterSet,
                                                  tns,
                                                  *preg_,
                                                  *act_table_,
                                                  actReg_,
                                                  processConfiguration(),
                                                  config,
                                                  processContext,
                                                  typeResolverMaker));
  }

  std::unique_ptr<Schedule> ScheduleItems::finishSchedule(MadeModules madeModules,
                                                          ParameterSet& parameterSet,
                                                          service::TriggerNamesService const& tns,
                                                          bool hasSubprocesses,
                                                          PreallocationConfiguration const& config,
                                                          ProcessContext const* processContext,
                                                          ProcessBlockHelperBase& processBlockHelper) {
    auto sched = std::move(madeModules.m_schedule);
    sched->finishSetup(parameterSet,
                       tns,
                       *preg_,
                       *branchIDListHelper_,
                       processBlockHelper,
                       *thinnedAssociationsHelper_,
                       subProcessParentageHelper_ ? subProcessParentageHelper_.get() : nullptr,
                       actReg_,
                       processConfiguration(),
                       hasSubprocesses,
                       config,
                       processContext);
    return sched;
  }

}  // namespace edm
