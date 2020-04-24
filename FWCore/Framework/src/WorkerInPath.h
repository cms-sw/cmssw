#ifndef FWCore_Framework_WorkerInPath_h
#define FWCore_Framework_WorkerInPath_h

/*

	Author: Jim Kowalkowski 28-01-06


	A wrapper around a Worker, so that statistics can be managed
	per path.  A Path holds Workers as these things.

*/

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"

namespace edm {

  class PathContext;
  class StreamID;
  class WaitingTask;

  class WorkerInPath {
  public:
    enum FilterAction { Normal=0, Ignore, Veto };

    WorkerInPath(Worker*, FilterAction theAction, unsigned int placeInPath);

    template <typename T>
    void runWorkerAsync(WaitingTask* iTask,
                        typename T::MyPrincipal const&, EventSetup const&,
                        StreamID streamID,
                        typename T::Context const* context);

    
    bool checkResultsOfRunWorker(bool wasEvent);
    
    void skipWorker(EventPrincipal const& iPrincipal) {
      worker_->skipOnPath();
    }
    void skipWorker(RunPrincipal const&) {}
    void skipWorker(LuminosityBlockPrincipal const&) {}
    
    void clearCounters() {
      timesVisited_ = timesPassed_ = timesFailed_ = timesExcept_ = 0;
    }
    
    int timesVisited() const { return timesVisited_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }

    FilterAction filterAction() const { return filterAction_; }
    Worker* getWorker() const { return worker_; }

    void setPathContext(PathContext const* v) { placeInPathContext_.setPathContext(v); }

  private:
    int timesVisited_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    
    FilterAction filterAction_;
    Worker* worker_;

    PlaceInPathContext placeInPathContext_;
  };
  
  inline bool WorkerInPath::checkResultsOfRunWorker(bool wasEvent) {
    if(not wasEvent) {
      return true;
    }
    auto state = worker_->state();
    bool rc = true;
    switch (state) {
      case Worker::Fail:
      {
        rc = false;
        break;
      }
      case Worker::Pass:
        break;
      case Worker::Exception:
      {
        ++timesExcept_;
        return true;
      }
        
      default:
        assert(false);
    }
    
    if(Ignore == filterAction()) {
      rc = true;
    } else if(Veto == filterAction()) {
      rc = !rc;
    }
    
    if(rc) {
      ++timesPassed_;
    } else {
      ++timesFailed_;
    }
    return rc;
    
  }

  template <typename T>
  void WorkerInPath::runWorkerAsync(WaitingTask* iTask,
                                    typename T::MyPrincipal const& ep, EventSetup const & es,
                                    StreamID streamID,
                                    typename T::Context const* context) {
    if (T::isEvent_) {
      ++timesVisited_;
    }
    
    if(T::isEvent_) {
      ParentContext parentContext(&placeInPathContext_);
      worker_->doWorkAsync<T>(iTask,ep, es,streamID, parentContext, context);
    } else {
      ParentContext parentContext(context);
      worker_->doWorkNoPrefetchingAsync<T>(iTask,ep, es,streamID, parentContext, context);
    }
  }  
}

#endif

