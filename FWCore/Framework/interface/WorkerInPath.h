#ifndef FWCore_Framework_WorkerInPath_h
#define FWCore_Framework_WorkerInPath_h

/*

	Author: Jim Kowalkowski 28-01-06


	A wrapper around a Worker, so that statistics can be managed
	per path.  A Path holds Workers as these things.

*/

#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"

#include <utility>

namespace edm {

  class PathContext;
  class StreamID;
  class ServiceToken;

  class WorkerInPath {
  public:
    enum FilterAction { Normal = 0, Ignore, Veto };

    WorkerInPath(Worker*, FilterAction theAction, unsigned int placeInPath, bool runConcurrently);

    void runWorkerAsync(
        WaitingTaskHolder, EventTransitionInfo const&, ServiceToken const&, StreamID, StreamContext const*) noexcept;

    bool checkResultsOfRunWorker();

    void skipWorker(EventPrincipal const& iPrincipal) { worker_->skipOnPath(iPrincipal); }

    FilterAction filterAction() const { return filterAction_; }
    Worker* getWorker() const { return worker_; }
    bool runConcurrently() const noexcept { return runConcurrently_; }
    unsigned int bitPosition() const noexcept { return placeInPathContext_.placeInPath(); }

    void setPathContext(PathContext const* v) { placeInPathContext_.setPathContext(v); }

  private:
    FilterAction filterAction_;
    Worker* worker_;

    PlaceInPathContext placeInPathContext_;
    bool runConcurrently_;
  };

  inline bool WorkerInPath::checkResultsOfRunWorker() {
    auto state = worker_->state();
    bool rc = true;
    switch (state) {
      case Worker::Fail: {
        rc = false;
        break;
      }
      case Worker::Pass:
        break;
      case Worker::Exception: {
        return true;
      }

      default:
        assert(false);
    }

    if (Ignore == filterAction()) {
      rc = true;
    } else if (Veto == filterAction()) {
      rc = !rc;
    }

    return rc;
  }

  inline void WorkerInPath::runWorkerAsync(WaitingTaskHolder iTask,
                                           EventTransitionInfo const& info,
                                           ServiceToken const& token,
                                           StreamID streamID,
                                           StreamContext const* context) noexcept {
    ParentContext parentContext(&placeInPathContext_);
    worker_->doWorkAsync<OccurrenceTraits<EventPrincipal, TransitionActionGlobalBegin>>(
        std::move(iTask), info, token, streamID, parentContext, context);
  }
}  // namespace edm

#endif
