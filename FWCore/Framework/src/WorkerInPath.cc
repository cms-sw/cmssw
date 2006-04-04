

#include "FWCore/Framework/src/WorkerInPath.h"

namespace edm
{
  WorkerInPath::WorkerInPath(Worker* w, State state):
    stopwatch_(new TStopwatch),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(state),
    worker_(w)
  {
    stopwatch_->Stop();
  }

  WorkerInPath::WorkerInPath(Worker* w):
    stopwatch_(new TStopwatch),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(Normal),
    worker_(w)
  {
    stopwatch_->Stop();
  }

  bool WorkerInPath::runWorker(EventPrincipal& ep, EventSetup const & es)
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
	rc = worker_->doWork(ep,es);

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
