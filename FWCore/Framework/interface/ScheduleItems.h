#ifndef FWCore_Framework_ScheduleItems_h
#define FWCore_Framework_ScheduleItems_h

#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <vector>

namespace edm {
  class ExceptionToActionTable;
  class ActivityRegistry;
  class BranchIDListHelper;
  class ThinnedAssociationsHelper;
  struct CommonParams;
  class SubProcess;
  class ParameterSet;
  class ProcessConfiguration;
  class ProcessContext;
  class ProductRegistry;
  class SignallingProductRegistry;
  class StreamID;
  class PreallocationConfiguration;
  class SubProcessParentageHelper;
  class ModuleTypeResolverMaker;
  namespace service {
    class TriggerNamesService;
  }

  struct ScheduleItems {
    ScheduleItems();

    ScheduleItems(ProductRegistry const& preg,
                  SubProcess const& om,
                  SubProcessBlockHelper& subProcessBlockHelper,
                  ProcessBlockHelperBase const& parentProcessBlockHelper);

    ScheduleItems(ScheduleItems const&) = delete;             // Disallow copying and moving
    ScheduleItems& operator=(ScheduleItems const&) = delete;  // Disallow copying and moving

    ServiceToken initServices(std::vector<ParameterSet>& servicePSets,
                              ParameterSet& processPSet,
                              ServiceToken const& iToken,
                              serviceregistry::ServiceLegacy iLegacy,
                              bool associate);

    ServiceToken addCPRandTNS(ParameterSet const& parameterSet, ServiceToken const& token);

    std::shared_ptr<CommonParams> initMisc(ParameterSet& parameterSet);

    std::unique_ptr<Schedule> initSchedule(ParameterSet& parameterSet,
                                           bool hasSubprocesses,
                                           PreallocationConfiguration const& iAllocConfig,
                                           ProcessContext const*,
                                           ModuleTypeResolverMaker const*,
                                           ProcessBlockHelperBase& processBlockHelper);

    class MadeModules {
      friend struct ScheduleItems;
      explicit MadeModules(std::unique_ptr<Schedule> iSched) : m_schedule(std::move(iSched)) {}

      std::unique_ptr<Schedule> m_schedule;

    public:
      MadeModules() = delete;
    };

    MadeModules initModules(ParameterSet& parameterSet,
                            service::TriggerNamesService const& tns,
                            PreallocationConfiguration const& iAllocConfig,
                            ProcessContext const*,
                            ModuleTypeResolverMaker const* typeResolverMaker);
    std::unique_ptr<Schedule> finishSchedule(MadeModules,
                                             ParameterSet& parameterSet,
                                             service::TriggerNamesService const& tns,
                                             bool hasSubprocesses,
                                             PreallocationConfiguration const& iAllocConfig,
                                             ProcessContext const*,
                                             ProcessBlockHelperBase& processBlockHelper);

    std::shared_ptr<SignallingProductRegistry const> preg() const { return get_underlying_safe(preg_); }
    std::shared_ptr<SignallingProductRegistry>& preg() { return get_underlying_safe(preg_); }
    std::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {
      return get_underlying_safe(branchIDListHelper_);
    }
    std::shared_ptr<BranchIDListHelper>& branchIDListHelper() { return get_underlying_safe(branchIDListHelper_); }
    std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper() const {
      return get_underlying_safe(thinnedAssociationsHelper_);
    }
    std::shared_ptr<ThinnedAssociationsHelper>& thinnedAssociationsHelper() {
      return get_underlying_safe(thinnedAssociationsHelper_);
    }
    std::shared_ptr<SubProcessParentageHelper>& subProcessParentageHelper() {
      return get_underlying_safe(subProcessParentageHelper_);
    }
    std::shared_ptr<ProcessConfiguration const> processConfiguration() const {
      return get_underlying_safe(processConfiguration_);
    }
    std::shared_ptr<ProcessConfiguration>& processConfiguration() { return get_underlying_safe(processConfiguration_); }

    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
    edm::propagate_const<std::shared_ptr<SignallingProductRegistry>> preg_;
    edm::propagate_const<std::shared_ptr<BranchIDListHelper>> branchIDListHelper_;
    edm::propagate_const<std::shared_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    edm::propagate_const<std::shared_ptr<SubProcessParentageHelper>> subProcessParentageHelper_;
    std::unique_ptr<ExceptionToActionTable const> act_table_;
    edm::propagate_const<std::shared_ptr<ProcessConfiguration>> processConfiguration_;
  };
}  // namespace edm
#endif
