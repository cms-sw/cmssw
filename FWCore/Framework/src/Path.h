#ifndef Framework_Path_h
#define Framework_Path_h

/*

  Author: Jim Kowalkowski 28-01-06

  $Id: Path.h,v 1.9 2006/06/20 23:13:27 paterno Exp $

  An object of this type represents one path in a job configuration.
  It holds the assigned bit position and the list of workers that are
  an event must pass through when this parh is processed.  The workers
  are held in WorkerInPath wrappers so that per path execution statistics
  can be kept for each worker.

*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/Worker.h"
#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "boost/shared_ptr.hpp"
#include "boost/signal.hpp"

#include <string>
#include <vector>

#include "FWCore/Framework/src/RunStopwatch.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm
{

  class Path
  {
  public:
    typedef edm::hlt::HLTState State;

    typedef std::vector<WorkerInPath> Workers;
    typedef Workers::size_type        size_type;
    typedef boost::shared_ptr<HLTGlobalStatus> TrigResPtr;
    typedef boost::shared_ptr<ActivityRegistry> ActivityRegistryPtr;

    Path(int bitpos, const std::string& path_name,
	 const Workers& workers,
	 TrigResPtr trptr,
	 ParameterSet const& proc_pset,
	 ActionTable& actions,
	 ActivityRegistryPtr reg);

    void runOneEvent(EventPrincipal&, EventSetup const&, BranchActionType const&);

    int bitPosition() const { return bitpos_; }
    const std::string& name() const { return name_; }

    std::pair<double,double> timeCpuReal() const {
      return std::pair<double,double>(stopwatch_->cpuTime(),stopwatch_->realTime());
    }

    std::pair<double,double> timeCpuReal(const unsigned int i) const {
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
    Worker const * getWorker(size_type i) const { return workers_.at(i).getWorker(); }

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

    Workers workers_;

    // Helper functions
    // nwrwue = numWorkersRunWithoutUnhandledException (really!)
    bool handleWorkerFailure(cms::Exception const& e, int nwrwue, bool isEvent);
    void recordUnknownException(int nwrwue, bool isEvent);
    void recordStatus(int nwrwue, bool isEvent);
    void updateCounters(bool succeed, bool isEvent);
  };
}

#endif
