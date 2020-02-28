#ifndef FWCore_Framework_GlobalSchedule_h
#define FWCore_Framework_GlobalSchedule_h

/*
*/

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include "boost/range/adaptor/reversed.hpp"

namespace edm {

  namespace {
    template <typename T>
    class GlobalScheduleSignalSentry {
    public:
      GlobalScheduleSignalSentry(ActivityRegistry* a, typename T::Context const* context)
          : a_(a), context_(context), allowThrow_(false) {
        if (a_)
          T::preScheduleSignal(a_, context_);
      }
      ~GlobalScheduleSignalSentry() noexcept(false) {
        // Caught exception is rethrown
        CMS_SA_ALLOW try {
          if (a_)
            T::postScheduleSignal(a_, context_);
        } catch (...) {
          if (allowThrow_) {
            throw;
          }
        }
      }

      void allowThrow() { allowThrow_ = true; }

    private:
      // We own none of these resources.
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      typename T::Context const* context_;
      bool allowThrow_;
    };
  }  // namespace

  class ActivityRegistry;
  class EventSetupImpl;
  class ExceptionCollector;
  class ProcessContext;
  class PreallocationConfiguration;
  class ModuleRegistry;
  class TriggerResultInserter;
  class PathStatusInserter;
  class EndPathStatusInserter;

  class GlobalSchedule {
  public:
    typedef std::vector<std::string> vstring;
    typedef std::vector<Worker*> AllWorkers;
    typedef std::shared_ptr<Worker> WorkerPtr;
    typedef std::vector<Worker*> Workers;

    GlobalSchedule(std::shared_ptr<TriggerResultInserter> inserter,
                   std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
                   std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
                   std::shared_ptr<ModuleRegistry> modReg,
                   std::vector<std::string> const& modulesToUse,
                   ParameterSet& proc_pset,
                   ProductRegistry& pregistry,
                   PreallocationConfiguration const& prealloc,
                   ExceptionToActionTable const& actions,
                   std::shared_ptr<ActivityRegistry> areg,
                   std::shared_ptr<ProcessConfiguration> processConfiguration,
                   ProcessContext const* processContext);
    GlobalSchedule(GlobalSchedule const&) = delete;

    template <typename T>
    void processOneGlobalAsync(WaitingTaskHolder holder,
                               typename T::MyPrincipal& principal,
                               EventSetupImpl const& eventSetup,
                               ServiceToken const& token,
                               bool cleaningUpAfterException = false);

    void beginJob(ProductRegistry const&, eventsetup::ESRecordsToProxyIndices const&);
    void endJob(ExceptionCollector& collector);

    /// Return a vector allowing const access to all the
    /// ModuleDescriptions for this GlobalSchedule.

    /// *** N.B. *** Ownership of the ModuleDescriptions is *not*
    /// *** passed to the caller. Do not call delete on these
    /// *** pointers!
    std::vector<ModuleDescription const*> getAllModuleDescriptions() const;

    /// Return the trigger report information on paths,
    /// modules-in-path, modules-in-endpath, and modules.
    void getTriggerReport(TriggerReport& rep) const;

    /// Return whether each output module has reached its maximum count.
    bool terminate() const;

    /// clone the type of module with label iLabel but configure with iPSet.
    void replaceModule(maker::ModuleHolder* iMod, std::string const& iLabel);

    /// returns the collection of pointers to workers
    AllWorkers const& allWorkers() const { return workerManagers_[0].allWorkers(); }

  private:
    //Sentry class to only send a signal if an
    // exception occurs. An exception is identified
    // by the destructor being called without first
    // calling completedSuccessfully().
    class SendTerminationSignalIfException {
    public:
      SendTerminationSignalIfException(edm::ActivityRegistry* iReg, edm::GlobalContext const* iContext)
          : reg_(iReg), context_(iContext) {}
      ~SendTerminationSignalIfException() {
        if (reg_) {
          reg_->preGlobalEarlyTerminationSignal_(*context_, TerminationOrigin::ExceptionFromThisContext);
        }
      }
      void completedSuccessfully() { reg_ = nullptr; }

    private:
      edm::ActivityRegistry* reg_;  // We do not use propagate_const because the registry itself is mutable.
      GlobalContext const* context_;
    };

    /// returns the action table
    ExceptionToActionTable const& actionTable() const { return workerManagers_[0].actionTable(); }

    std::vector<WorkerManager> workerManagers_;
    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
    std::vector<edm::propagate_const<WorkerPtr>> extraWorkers_;
    ProcessContext const* processContext_;
  };

  template <typename T>
  void GlobalSchedule::processOneGlobalAsync(WaitingTaskHolder iHolder,
                                             typename T::MyPrincipal& ep,
                                             EventSetupImpl const& es,
                                             ServiceToken const& token,
                                             bool cleaningUpAfterException) {
    // Caught exception is propagated via WaitingTaskHolder
    CMS_SA_ALLOW try {
      //need the doneTask to own the memory
      auto globalContext = std::make_shared<GlobalContext>(T::makeGlobalContext(ep, processContext_));

      if (actReg_) {
        //Services may depend upon each other
        ServiceRegistry::Operate op(token);
        T::preScheduleSignal(actReg_.get(), globalContext.get());
      }

      auto doneTask = make_waiting_task(
          tbb::task::allocate_root(),
          [this, iHolder, cleaningUpAfterException, globalContext, token](std::exception_ptr const* iPtr) mutable {
            std::exception_ptr excpt;
            if (iPtr) {
              excpt = *iPtr;
              //add context information to the exception and print message
              try {
                convertException::wrap([&]() { std::rethrow_exception(excpt); });
              } catch (cms::Exception& ex) {
                //TODO: should add the transition type info
                std::ostringstream ost;
                if (ex.context().empty()) {
                  ost << "Processing " << T::transitionName() << " ";
                }
                ServiceRegistry::Operate op(token);
                addContextAndPrintException(ost.str().c_str(), ex, cleaningUpAfterException);
                excpt = std::current_exception();
              }
              if (actReg_) {
                ServiceRegistry::Operate op(token);
                actReg_->preGlobalEarlyTerminationSignal_(*globalContext, TerminationOrigin::ExceptionFromThisContext);
              }
            }
            if (actReg_) {
              // Caught exception is propagated via WaitingTaskHolder
              CMS_SA_ALLOW try {
                ServiceRegistry::Operate op(token);
                T::postScheduleSignal(actReg_.get(), globalContext.get());
              } catch (...) {
                if (not excpt) {
                  excpt = std::current_exception();
                }
              }
            }
            iHolder.doneWaiting(excpt);
          });
      workerManagers_[ep.index()].resetAll();

      ParentContext parentContext(globalContext.get());
      //make sure the ProductResolvers know about their
      // workers to allow proper data dependency handling
      workerManagers_[ep.index()].setupOnDemandSystem(ep, es);

      //make sure the task doesn't get run until all workers have beens started
      WaitingTaskHolder holdForLoop(doneTask);
      auto& aw = workerManagers_[ep.index()].allWorkers();
      for (Worker* worker : boost::adaptors::reverse(aw)) {
        worker->doWorkAsync<T>(
            doneTask, ep, es, token, StreamID::invalidStreamID(), parentContext, globalContext.get());
      }
    } catch (...) {
      iHolder.doneWaiting(std::current_exception());
    }
  }
}  // namespace edm

#endif
