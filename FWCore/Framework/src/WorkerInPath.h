#ifndef FWCore_Framework_WorkerInPath_h
#define FWCore_Framework_WorkerInPath_h

/*

	Author: Jim Kowalkowski 28-01-06

	$Id: WorkerInPath.h,v 1.12 2008/10/16 23:06:28 wmtan Exp $

	A wrapper around a Worker, so that statistics can be managed
	per path.  A Path holds Workers as these things.

*/

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/RunStopwatch.h"

namespace edm {

  class WorkerInPath {
  public:
    enum FilterAction { Normal=0, Ignore, Veto };

    explicit WorkerInPath(Worker*);
    WorkerInPath(Worker*, FilterAction theAction);

    template <typename T>
    bool runWorker(typename T::MyPrincipal&, EventSetup const&,
		   CurrentProcessingContext const* cpc);

    std::pair<double,double> timeCpuReal() const {
      return std::pair<double,double>(stopwatch_->cpuTime(),stopwatch_->realTime());
    }

    void clearCounters() {
      timesVisited_ = timesPassed_ = timesFailed_ = timesExcept_ = 0;
    }

    int timesVisited() const { return timesVisited_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }

    FilterAction filterAction() const { return filterAction_; }
    Worker* getWorker() const { return worker_; }

  private:
    RunStopwatch::StopwatchPointer stopwatch_;

    int timesVisited_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    
    FilterAction filterAction_;
    Worker* worker_;
  };

  template <typename T>
  bool WorkerInPath::runWorker(typename T::MyPrincipal & ep, EventSetup const & es,
			       CurrentProcessingContext const* cpc) {

    // A RunStopwatch, but only if we are processing an event.
    std::auto_ptr<RunStopwatch> stopwatch(T::isEvent_ ? new RunStopwatch(stopwatch_) : 0);

    if (T::isEvent_) {
      ++timesVisited_;
    }
    bool rc = true;

    try {
	// may want to change the return value from the worker to be 
	// the Worker::FilterAction so conditions in the path will be easier to 
	// identify
	rc = worker_->doWork<T>(ep, es, cpc);

        // Ignore return code for non-event (e.g. run, lumi) calls
	if (!T::isEvent_) rc = true;
	else if (filterAction_ == Veto) rc = !rc;
        else if (filterAction_ == Ignore) rc = true;

	if (T::isEvent_) {
	  if(rc) ++timesPassed_; else ++timesFailed_;
	}
    }
    catch(...) {
	if (T::isEvent_) ++timesExcept_;
	throw;
    }

    return rc;
  }

}

#endif

