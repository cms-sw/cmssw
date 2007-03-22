

#include "FWCore/Framework/src/WorkerInPath.h"

namespace edm
{
  WorkerInPath::WorkerInPath(Worker* w, FilterAction theFilterAction):
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    filterAction_(theFilterAction),
    worker_(w)
  {
  }

  WorkerInPath::WorkerInPath(Worker* w):
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    filterAction_(Normal),
    worker_(w)
  {
  }

  bool WorkerInPath::runWorker(EventPrincipal& ep, EventSetup const & es,
			       BranchActionType const& bat,
			       CurrentProcessingContext const* cpc)
  {
    bool const isEvent = (bat == BranchActionEvent);

    // A RunStopwatch, but only if we are processing an event.
    std::auto_ptr<RunStopwatch> stopwatch(isEvent ? new RunStopwatch(stopwatch_) : 0);

    if (isEvent) {
      ++timesVisited_;
    }
    bool rc = true;

    try {
	// may want to change the return value from the worker to be 
	// the Worker::FilterAction so conditions in the path will be easier to 
	// identify
	rc = worker_->doWork(ep, es, bat, cpc);

	if (filterAction_ == Veto) rc = !rc;
        else if (filterAction_ == Ignore) rc = true;

	if (isEvent) {
	  if(rc) ++timesPassed_; else ++timesFailed_;
	}
    }
    catch(...) {
	if (isEvent) ++timesExcept_;
	throw;
    }

    return rc;
  }

}
