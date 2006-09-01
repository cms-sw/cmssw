

#include "FWCore/Framework/src/WorkerInPath.h"

namespace edm
{
  WorkerInPath::WorkerInPath(Worker* w, State theState):
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(theState),
    worker_(w)
  {
  }

  WorkerInPath::WorkerInPath(Worker* w):
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(Normal),
    worker_(w)
  {
  }

  bool WorkerInPath::runWorker(EventPrincipal& ep, EventSetup const & es,
			       CurrentProcessingContext const* cpc)
  {
    RunStopwatch stopwatch(stopwatch_);
    ++timesVisited_;
    bool rc = true;

    if(state_ == Ignore)
      {
	// ++timesPassed_; // should this be incremented or not?
	return rc;
      }

    try 
      {
	// may want to change the return value from the worker to be 
	// the Worker::State so conditions in the path will be easier to 
	// identify
	rc = worker_->doWork(ep,es,cpc);

	if(state_ == Veto) rc = !rc;

	if(rc) ++timesPassed_; else ++timesFailed_;
      }
    catch(...)
      {
	++timesExcept_;
	throw;
      }

    return rc;
  }

}
