#ifndef FWCore_Framework_UnscheduledCallProducer_h
#define FWCore_Framework_UnscheduledCallProducer_h

// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     UnscheduledCallProducer
//
/**\class UnscheduledCallProducer UnscheduledCallProducer.h "UnscheduledCallProducer.h"

 Description: Handles calling of EDProducers which are unscheduled

 Usage:
 <usage>

 */

#include "FWCore/Framework/interface/BranchActionType.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/UnscheduledAuxiliary.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <cassert>

namespace edm {

  class EventTransitionInfo;
  class ModuleCallingContext;

  class UnscheduledCallProducer {
  public:
    using worker_container = std::vector<Worker*>;
    using const_iterator = worker_container::const_iterator;

    UnscheduledCallProducer(ActivityRegistry& iReg) : unscheduledWorkers_() {
      aux_.preModuleDelayedGetSignal_.connect(std::cref(iReg.preModuleEventDelayedGetSignal_));
      aux_.postModuleDelayedGetSignal_.connect(std::cref(iReg.postModuleEventDelayedGetSignal_));
    }
    void addWorker(Worker* aWorker) {
      assert(nullptr != aWorker);
      unscheduledWorkers_.push_back(aWorker);
      if (aWorker->hasAccumulator()) {
        accumulatorWorkers_.push_back(aWorker);
      }
    }

    void removeWorker(Worker const* worker) {
      unscheduledWorkers_.erase(std::remove(unscheduledWorkers_.begin(), unscheduledWorkers_.end(), worker),
                                unscheduledWorkers_.end());
      accumulatorWorkers_.erase(std::remove(accumulatorWorkers_.begin(), accumulatorWorkers_.end(), worker),
                                accumulatorWorkers_.end());
    }

    void setEventTransitionInfo(EventTransitionInfo const& info) { aux_.setEventTransitionInfo(info); }

    UnscheduledAuxiliary const& auxiliary() const { return aux_; }

    const_iterator begin() const { return unscheduledWorkers_.begin(); }
    const_iterator end() const { return unscheduledWorkers_.end(); }

    template <typename T, typename U>
    void runNowAsync(WaitingTaskHolder task,
                     typename T::TransitionInfoType const& info,
                     ServiceToken const& token,
                     StreamID streamID,
                     typename T::Context const* topContext,
                     U const* context) const {
      //do nothing for event since we will run when requested
      if (!T::isEvent_) {
        for (auto worker : unscheduledWorkers_) {
          ParentContext parentContext(context);

          // We do not need to run prefetching here because this only handles
          // stream transitions for runs and lumis. There are no products put
          // into the runs or lumis in stream transitions, so there can be
          // no data dependencies which require prefetching. Prefetching is
          // needed for global transitions, but they are run elsewhere.
          worker->doWorkNoPrefetchingAsync<T>(task, info, token, streamID, parentContext, topContext);
        }
      }
    }

    template <typename T>
    void runAccumulatorsAsync(WaitingTaskHolder task,
                              typename T::TransitionInfoType const& info,
                              ServiceToken const& token,
                              StreamID streamID,
                              ParentContext const& parentContext,
                              typename T::Context const* context) {
      for (auto worker : accumulatorWorkers_) {
        worker->doWorkAsync<T>(task, info, token, streamID, parentContext, context);
      }
    }

  private:
    template <typename T, typename ID>
    void addContextToException(cms::Exception& ex, Worker const* worker, ID const& id) const {
      std::ostringstream ost;
      ost << "Processing " << T::transitionName() << " " << id;
      ex.addContext(ost.str());
    }
    worker_container unscheduledWorkers_;
    worker_container accumulatorWorkers_;
    UnscheduledAuxiliary aux_;
  };

}  // namespace edm

#endif
