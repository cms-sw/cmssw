#ifndef Framework_Worker_h
#define Framework_Worker_h

/*----------------------------------------------------------------------
  
Worker: this is a basic scheduling unit - an abstract base class to
something that is really a producer or filter.

$Id: Worker.h,v 1.13 2006/06/20 23:13:27 paterno Exp $

A worker will not actually call through to the module unless it is
in a Ready state.  After a module is actually run, the state will not
be Ready.  The Ready state can only be reestablished by doing a reset().

Pre/post module signals are posted only in the Ready state.

Execution statistics are kept here.

If a module has thrown an exception during execution, that exception
will be rethrown if the worker is entered again and the state is not Ready.
In other words, execution results (status) are cached and reused until
the worker is reset().

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/RunStopwatch.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class Worker
  {
  public:
    enum State { Ready, Pass, Fail, Exception };

    Worker(const ModuleDescription& iMD, const WorkerParams& iWP);
    virtual ~Worker();

    bool doWork(EventPrincipal&, EventSetup const& c,
		CurrentProcessingContext const* cpc) ;
    void beginJob(EventSetup const&) ;
    void endJob();
    void reset() { state_ = Ready; }
    
    ModuleDescription const & description() const {return md_;}
    ModuleDescription const * descPtr() const {return &md_; }
    void connect(ActivityRegistry::PreModule&, ActivityRegistry::PostModule&);

    std::pair<double,double> timeCpuReal() const {
      return std::pair<double,double>(stopwatch_->cpuTime(),stopwatch_->realTime());
    }

    int timesRun() const { return timesRun_; }
    int timesVisited() const { return timesVisited_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }
    State state() const { return state_; }
   
    struct Sigs
    {
      ActivityRegistry::PreModule preModuleSignal;
      ActivityRegistry::PostModule postModuleSignal;
    };

    int timesPass() const { return timesPassed(); } // for backward compatibility only - to be removed soon

  protected:
    virtual std::string workerType() const = 0;
    virtual bool implDoWork(EventPrincipal&, EventSetup const& c, 
			    CurrentProcessingContext const* cpc) = 0;
    virtual void implBeginJob(EventSetup const&) = 0;
    virtual void implEndJob() = 0;

  private:
    RunStopwatch::StopwatchPointer stopwatch_;

    int timesRun_;
    int timesVisited_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    State state_;

    ModuleDescription md_;
    const ActionTable* actions_; // memory assumed to be managed elsewhere
    boost::shared_ptr<cms::Exception> cached_exception_; // if state is 'exception'

    Sigs sigs_;

  };

  template <class WT>
  struct WorkerType
  {
    // typedef int module_type;
    // typedef int worker_type;
  };

}
#endif
