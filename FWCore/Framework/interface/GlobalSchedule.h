#ifndef FWCore_Framework_GlobalSchedule_h
#define FWCore_Framework_GlobalSchedule_h

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/WorkerRegistry.h"
#include "FWCore/Framework/interface/SignallingProductRegistry.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <exception>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include "boost/range/adaptor/reversed.hpp"

namespace edm {

  class ExceptionCollector;
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
                   SignallingProductRegistry& pregistry,
                   PreallocationConfiguration const& prealloc,
                   ExceptionToActionTable const& actions,
                   std::shared_ptr<ActivityRegistry> areg,
                   std::shared_ptr<ProcessConfiguration const> processConfiguration,
                   ProcessContext const* processContext);
    GlobalSchedule(GlobalSchedule const&) = delete;

    template <typename T>
    void processOneGlobalAsync(WaitingTaskHolder holder,
                               typename T::TransitionInfoType&,
                               ServiceToken const& token,
                               bool cleaningUpAfterException = false);

    void beginJob(ProductRegistry const&,
                  eventsetup::ESRecordsToProductResolverIndices const&,
                  ProcessBlockHelperBase const&,
                  PathsAndConsumesOfModulesBase const&,
                  ProcessContext const&);
    void endJob(ExceptionCollector& collector);

    /// Return a vector allowing const access to all the
    /// ModuleDescriptions for this GlobalSchedule.

    /// *** N.B. *** Ownership of the ModuleDescriptions is *not*
    /// *** passed to the caller. Do not call delete on these
    /// *** pointers!
    std::vector<ModuleDescription const*> getAllModuleDescriptions() const;

    /// Return whether each output module has reached its maximum count.
    bool terminate() const;

    /// clone the type of module with label iLabel but configure with iPSet.
    void replaceModule(maker::ModuleHolder* iMod, std::string const& iLabel);

    /// Delete the module with label iLabel
    void deleteModule(std::string const& iLabel);

    /// returns the collection of pointers to workers
    AllWorkers const& allWorkers() const { return workerManagers_[0].allWorkers(); }

    void releaseMemoryPostLookupSignal();

  private:
    /// returns the action table
    ExceptionToActionTable const& actionTable() const { return workerManagers_[0].actionTable(); }

    template <typename T>
    void preScheduleSignal(GlobalContext const*, ServiceToken const&);

    template <typename T>
    void postScheduleSignal(GlobalContext const*, ServiceWeakToken const&, std::exception_ptr&);

    void handleException(GlobalContext const*,
                         ServiceWeakToken const&,
                         bool cleaningUpAfterException,
                         std::exception_ptr&);

    std::vector<WorkerManager> workerManagers_;
    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
    std::vector<edm::propagate_const<WorkerPtr>> extraWorkers_;
    ProcessContext const* processContext_;

    // The next 4 variables use the same naming convention, even though we have no intention
    // to ever have concurrent ProcessBlocks or Jobs. They are all related to the number of
    // WorkerManagers needed for global transitions.
    unsigned int numberOfConcurrentLumis_;
    unsigned int numberOfConcurrentRuns_;
    static constexpr unsigned int numberOfConcurrentProcessBlocks_ = 1;
    static constexpr unsigned int numberOfConcurrentJobs_ = 1;
  };

  template <typename T>
  void GlobalSchedule::processOneGlobalAsync(WaitingTaskHolder iHolder,
                                             typename T::TransitionInfoType& transitionInfo,
                                             ServiceToken const& token,
                                             bool cleaningUpAfterException) {
    auto const& principal = transitionInfo.principal();

    // Caught exception is propagated via WaitingTaskHolder
    CMS_SA_ALLOW try {
      //need the doneTask to own the memory
      auto globalContext = std::make_shared<GlobalContext>(T::makeGlobalContext(principal, processContext_));

      ServiceWeakToken weakToken = token;
      auto doneTask = make_waiting_task(
          [this, iHolder, cleaningUpAfterException, globalContext, weakToken](std::exception_ptr const* iPtr) mutable {
            std::exception_ptr excpt;
            if (iPtr) {
              excpt = *iPtr;
              // add context information to the exception and print message
              handleException(globalContext.get(), weakToken, cleaningUpAfterException, excpt);
            }
            postScheduleSignal<T>(globalContext.get(), weakToken, excpt);
            iHolder.doneWaiting(excpt);
          });

      //make sure the task doesn't get run until all workers have beens started
      WaitingTaskHolder holdForLoop(*iHolder.group(), doneTask);

      CMS_SA_ALLOW try {
        preScheduleSignal<T>(globalContext.get(), token);

        unsigned int managerIndex = principal.index();
        if constexpr (T::branchType_ == InRun) {
          managerIndex += numberOfConcurrentLumis_;
        } else if constexpr (T::branchType_ == InProcess) {
          managerIndex += (numberOfConcurrentLumis_ + numberOfConcurrentRuns_);
        }
        WorkerManager& workerManager = workerManagers_[managerIndex];
        workerManager.resetAll();

        ParentContext parentContext(globalContext.get());
        // make sure the ProductResolvers know about their
        // workers to allow proper data dependency handling
        workerManager.setupResolvers(transitionInfo.principal());

        auto& aw = workerManager.allWorkers();
        for (Worker* worker : boost::adaptors::reverse(aw)) {
          worker->doWorkAsync<T>(
              holdForLoop, transitionInfo, token, StreamID::invalidStreamID(), parentContext, globalContext.get());
        }
      } catch (...) {
        holdForLoop.doneWaiting(std::current_exception());
      }
    } catch (...) {
      iHolder.doneWaiting(std::current_exception());
    }
  }

  template <typename T>
  void GlobalSchedule::preScheduleSignal(GlobalContext const* globalContext, ServiceToken const& token) {
    if (actReg_) {
      try {
        ServiceRegistry::Operate op(token);
        convertException::wrap([this, globalContext]() { T::preScheduleSignal(actReg_.get(), globalContext); });
      } catch (cms::Exception& ex) {
        exceptionContext(ex, *globalContext, "Handling pre signal, likely in a service function");
        throw;
      }
    }
  }

  template <typename T>
  void GlobalSchedule::postScheduleSignal(GlobalContext const* globalContext,
                                          ServiceWeakToken const& weakToken,
                                          std::exception_ptr& excpt) {
    if (actReg_) {
      try {
        convertException::wrap([this, &weakToken, globalContext]() {
          ServiceRegistry::Operate op(weakToken.lock());
          T::postScheduleSignal(actReg_.get(), globalContext);
        });
      } catch (cms::Exception& ex) {
        if (not excpt) {
          exceptionContext(ex, *globalContext, "Handling post signal, likely in a service function");
          excpt = std::current_exception();
        }
      }
    }
  }

}  // namespace edm

#endif
