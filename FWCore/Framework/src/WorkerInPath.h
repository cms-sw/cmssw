#ifndef Framework_WorkerInPath_h
#define Framework_WorkerInPath_h

/*

	Author: Jim Kowalkowski 28-01-06

	$Id: WorkerInPath.h,v 1.1 2006/01/29 23:33:58 jbk Exp $

	A wrapper around a Worker, so that statistics can be managed
	per path.  A Path holds Workers as these things.

*/

#include "FWCore/Framework/src/Worker.h"

#include "boost/shared_ptr.hpp"

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

    int timesVisited() const { return timesVisited_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }

    Worker* getWorker() { return worker_; }

  private:
    int timesVisited_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    
    State state_;
    Worker* worker_;
  };
}

#endif

