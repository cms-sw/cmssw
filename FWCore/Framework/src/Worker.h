#ifndef Framework_Worker_h
#define Framework_Worker_h

/*----------------------------------------------------------------------
  
Worker: this is a basic scheduling unit - an abstract base class to
something that is really a producer or filter.

$Id: Worker.h,v 1.7 2005/12/28 00:21:58 wmtan Exp $

A worker will not actually call through to the module unless it is
in a Ready state.  After a module is actually run, the state will not
be Ready.  The Ready state can only be reestablished by doing a reset().

Pre/post module signals are posted onyl in the ready state.

Execution statistics are kept here.

If a module has thrown an exception during execution, that exception
will be rethrown if the worked is entered again and the state is not Ready.
In other words, execution results (status) are cached and reused until
the worker in reset().

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "boost/shared_ptr.hpp"

namespace edm {
  class ActionTable;
  class EventPrincipal;
  class EventSetup;

  class Worker
  {
  public:
    enum State { Ready, Pass, Fail, Exception };

    Worker(const ModuleDescription& iMD, const WorkerParams& iWP);
    virtual ~Worker();

    bool doWork(EventPrincipal&, EventSetup const& c) ;
    void beginJob(EventSetup const&) ;
    void endJob();
    void reset() { state_ = Ready; }
    
    const ModuleDescription& description() const {return md_;}
    State state() const { return state_; }
    void connect(ActivityRegistry::PreModule&, ActivityRegistry::PostModule&);

    int timesVisited() const { return timesVisited_; }
    int timesRun() const { return timesRun_; }
    int timesFailed() const { return timesFailed_; }
    int timesPass() const { return timesPass_; }
    int timesExcept() const { return timesExcept_; }
   
    struct Sigs
    {
      boost::signal<void (const ModuleDescription&)> preModuleSignal;
      boost::signal<void (const ModuleDescription&)> postModuleSignal;
    };

  protected:
    virtual std::string workerType() const = 0;
    virtual bool implDoWork(EventPrincipal&, EventSetup const& c) = 0;
    virtual void implBeginJob(EventSetup const&) = 0;
    virtual void implEndJob() = 0;
    
  private:
    int timesVisited_;
    int timesRun_;
    int timesFailed_;
    int timesPass_;
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
