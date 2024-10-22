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

#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/UnscheduledAuxiliary.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

namespace edm {

  class EventTransitionInfo;

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
    worker_container const& workers() const { return unscheduledWorkers_; }

    template <typename T>
    void runAccumulatorsAsync(WaitingTaskHolder task,
                              typename T::TransitionInfoType const& info,
                              ServiceToken const& token,
                              StreamID streamID,
                              ParentContext const& parentContext,
                              typename T::Context const* context) noexcept {
      for (auto worker : accumulatorWorkers_) {
        worker->doWorkAsync<T>(task, info, token, streamID, parentContext, context);
      }
    }

  private:
    worker_container unscheduledWorkers_;
    worker_container accumulatorWorkers_;
    UnscheduledAuxiliary aux_;
  };

}  // namespace edm

#endif
