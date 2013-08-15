#ifndef FWCore_Framework_ScheduleItems_h
#define FWCore_Framework_ScheduleItems_h

#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "boost/shared_ptr.hpp"

#include <memory>
#include <vector>

namespace edm {
  class ActionTable;
  class ActivityRegistry;
  class BranchIDListHelper;
  class CommonParams;
  class SubProcess;
  class ParameterSet;
  class ProcessConfiguration;
  class ProcessContext;
  class ProductRegistry;
  class Schedule;
  class SignallingProductRegistry;
  class StreamID;

  struct ScheduleItems {
    ScheduleItems();

    ScheduleItems(ProductRegistry const& preg, BranchIDListHelper const& branchIDListHelper, SubProcess const& om);

    ScheduleItems(ScheduleItems const&) = delete; // Disallow copying and moving
    ScheduleItems& operator=(ScheduleItems const&) = delete; // Disallow copying and moving

    ServiceToken
    initServices(std::vector<ParameterSet>& servicePSets,
                 ParameterSet& processPSet,
                 ServiceToken const& iToken,
                 serviceregistry::ServiceLegacy iLegacy,
                 bool associate);

    ServiceToken
    addCPRandTNS(ParameterSet const& parameterSet, ServiceToken const& token);

    boost::shared_ptr<CommonParams>
    initMisc(ParameterSet& parameterSet);

    std::auto_ptr<Schedule>
    initSchedule(ParameterSet& parameterSet,
                 ParameterSet const* subProcessPSet,
                 StreamID streamID,
                 ProcessContext const*);

    void
    clear();

    boost::shared_ptr<ActivityRegistry>           actReg_;
    std::unique_ptr<SignallingProductRegistry>    preg_;
    boost::shared_ptr<BranchIDListHelper>         branchIDListHelper_;
    std::unique_ptr<ActionTable const>            act_table_;
    boost::shared_ptr<ProcessConfiguration>       processConfiguration_;
  };
}

#endif
