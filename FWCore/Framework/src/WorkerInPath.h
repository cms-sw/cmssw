#ifndef Framework_WorkerInPath_h
#define Framework_WorkerInPath_h

/*

	Author: Jim Kowalkowski 28-01-06

	$Id: WorkerInPath.h,v 1.2 2006/02/04 04:31:01 jbk Exp $

	A wrapper around a Worker, so that statistics can be managed
	per path.  A Path holds Workers as these things.

*/

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/RunStopwatch.h"

namespace edm
{
  class EventPrincipal;
  class EventSetup;

  class WorkerInPath
  {
  public:
    enum State { Normal=0, Ignore, Veto };

    explicit WorkerInPath(Worker*);
    WorkerInPath(Worker*, State state);

    bool runWorker(EventPrincipal&, EventSetup const&);

    std::pair<double,double> timeCpuReal() const {
      return std::pair<double,double>(stopwatch_->CpuTime(),stopwatch_->RealTime());
    }

    int timesVisited() const { return timesVisited_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }

    State state() const { return state_; }
    Worker* getWorker() const { return worker_; }

  private:
    RunStopwatch::StopwatchPointer stopwatch_;

    int timesVisited_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    
    State state_;
    Worker* worker_;
  };
}

#endif

