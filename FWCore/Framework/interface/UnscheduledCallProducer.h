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
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>

namespace edm {

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
    }
    
    void setEventSetup(EventSetup const& iSetup) {
      aux_.setEventSetup(&iSetup);
    }

    UnscheduledAuxiliary const& auxiliary() const { return aux_; }

    const_iterator begin() const { return unscheduledWorkers_.begin(); }
    const_iterator end() const { return unscheduledWorkers_.end(); }
    
    template <typename T, typename U>
    void runNowAsync(WaitingTask* task,
                     typename T::MyPrincipal& p, EventSetup const& es, StreamID streamID,
                     typename T::Context const* topContext, U const* context) const {
      //do nothing for event since we will run when requested
      if(!T::isEvent_) {
        for(auto worker: unscheduledWorkers_) {
          ParentContext parentContext(context);
          worker->doWorkNoPrefetchingAsync<T>(task, p, es, streamID, parentContext, topContext);
        }
      }
    }

    
  private:
    template <typename T, typename ID>
    void addContextToException(cms::Exception& ex, Worker const* worker, ID const& id) const {
      std::ostringstream ost;
      ost << "Processing " << T::transitionName()<<" "<< id;
      ex.addContext(ost.str());
    }
    worker_container unscheduledWorkers_;
    UnscheduledAuxiliary aux_;
  };

}

#endif

