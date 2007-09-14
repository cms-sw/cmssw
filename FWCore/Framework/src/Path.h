#ifndef FWCore_Framework_Path_h
#define FWCore_Framework_Path_h

/*

  Author: Jim Kowalkowski 28-01-06

  $Id: Path.h,v 1.14 2007/06/14 17:52:18 wmtan Exp $

  An object of this type represents one path in a job configuration.
  It holds the assigned bit position and the list of workers that are
  an event must pass through when this parh is processed.  The workers
  are held in WorkerInPath wrappers so that per path execution statistics
  can be kept for each worker.

*/

#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/Worker.h"
#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>

#include "FWCore/Framework/src/RunStopwatch.h"

namespace edm {
  class ParameterSet;

  class Path {
  public:
    typedef edm::hlt::HLTState State;

    typedef std::vector<WorkerInPath> WorkersInPath;
    typedef WorkersInPath::size_type        size_type;
    typedef boost::shared_ptr<HLTGlobalStatus> TrigResPtr;
    typedef boost::shared_ptr<ActivityRegistry> ActivityRegistryPtr;

    Path(int bitpos, std::string const& path_name,
	 WorkersInPath const& workers,
	 TrigResPtr trptr,
	 ParameterSet const& proc_pset,
	 ActionTable& actions,
	 ActivityRegistryPtr reg,
	 bool isEndPath);

    template <typename T>
    void runOneEvent(T&, EventSetup const&, BranchActionType const&);

    int bitPosition() const { return bitpos_; }
    std::string const& name() const { return name_; }

    std::pair<double,double> timeCpuReal() const {
      return std::pair<double,double>(stopwatch_->cpuTime(),stopwatch_->realTime());
    }

    std::pair<double,double> timeCpuReal(unsigned int const i) const {
      return workers_.at(i).timeCpuReal();
    }

    int timesRun() const { return timesRun_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }
    //int abortWorker() const { return abortWorker_; }
    State state() const { return state_; }

    size_type size() const { return workers_.size(); }
    int timesVisited(size_type i) const { return workers_.at(i).timesVisited(); }
    int timesPassed (size_type i) const { return workers_.at(i).timesPassed() ; }
    int timesFailed (size_type i) const { return workers_.at(i).timesFailed() ; }
    int timesExcept (size_type i) const { return workers_.at(i).timesExcept() ; }
    Worker const* getWorker(size_type i) const { return workers_.at(i).getWorker(); }

  private:
    RunStopwatch::StopwatchPointer stopwatch_;
    int timesRun_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    //int abortWorker_;
    State state_;

    int bitpos_;
    std::string name_;
    TrigResPtr trptr_;
    ActivityRegistryPtr act_reg_;
    ActionTable* act_table_;

    WorkersInPath workers_;

    bool isEndPath_;

    // Helper functions
    // nwrwue = numWorkersRunWithoutUnhandledException (really!)
    bool handleWorkerFailure(cms::Exception const& e, int nwrwue, bool isEvent);
    void recordUnknownException(int nwrwue, bool isEvent);
    void recordStatus(int nwrwue, bool isEvent);
    void updateCounters(bool succeed, bool isEvent);
  };

  namespace {
    struct PathSignalSentry {
      PathSignalSentry(std::string const& name,
                         int const& nwrwue,
                         edm::hlt::HLTState const& state,
                         edm::Path::ActivityRegistryPtr areg ):
      name_(name),
      nwrwue_(nwrwue),
      state_(state),
      areg_(areg) {
        areg_->preProcessPathSignal_(name_);
      }
      ~PathSignalSentry() {
        HLTPathStatus status(state_, nwrwue_);
        areg_->postProcessPathSignal_(name_, status);
      }
      std::string const& name_;
      int const& nwrwue_;
      edm::hlt::HLTState const& state_;
      edm::Path::ActivityRegistryPtr areg_;
    };
  }

  template <typename T>
  void Path::runOneEvent(T& ep,
	     EventSetup const& es,
	     BranchActionType const& bat) {
    bool const isEvent = (bat == BranchActionEvent);

    //Create the PathSignalSentry before the RunStopwatch so that
    // we only record the time spent in the path not from the signal
    int nwrwue = -1;
    std::auto_ptr<PathSignalSentry> signaler(isEvent? new PathSignalSentry(name_,
                                                                           nwrwue,
                                                                           state_,
                                                                           act_reg_) : 0 );
                                                                           
    // A RunStopwatch, but only if we are processing an event.
    std::auto_ptr<RunStopwatch> stopwatch(isEvent ? new RunStopwatch(stopwatch_) : 0);

    if (isEvent) {
      ++timesRun_;
    }
    state_ = edm::hlt::Ready;

    // nwrue =  numWorkersRunWithoutUnhandledException
    bool should_continue = true;
    CurrentProcessingContext cpc(&name_, bitPosition(), isEndPath_);

    WorkersInPath::size_type idx = 0;
    // It seems likely that 'nwrwue' and 'idx' can never differ ---
    // if so, we should remove one of them!.
    for (WorkersInPath::iterator i = workers_.begin(), end = workers_.end();
          i != end && should_continue;
          ++i, ++idx) {
      ++nwrwue;
      assert (static_cast<int>(idx) == nwrwue);
      try {
        cpc.activate(idx, i->getWorker()->descPtr());
        should_continue = i->runWorker(ep, es, bat, &cpc);
      }
      catch(cms::Exception& e) {
        // handleWorkerFailure may throw a new exception.
        should_continue = handleWorkerFailure(e, nwrwue, isEvent);
      }
      catch(...) {
        recordUnknownException(nwrwue, isEvent);
        throw;
      }
    }
    updateCounters(should_continue, isEvent);
    recordStatus(nwrwue, isEvent);
  }

}

#endif
