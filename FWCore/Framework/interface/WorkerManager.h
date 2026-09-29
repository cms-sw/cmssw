#ifndef FWCore_Framework_WorkerManager_h
#define FWCore_Framework_WorkerManager_h

#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/UnscheduledCallProducer.h"
#include "FWCore/Framework/interface/WorkerRegistry.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace edm {
  class ExceptionToActionTable;
  class ModuleRegistry;
  class Worker;
  namespace eventsetup {
    class ESRecordsToProductResolverIndices;
  }

  template <typename TI, typename TP>
  class WorkerManagerCore {
  public:
    typedef std::vector<Worker*> AllWorkers;
    WorkerManagerCore(WorkerManagerCore&&) = default;

    WorkerManagerCore(std::shared_ptr<ModuleRegistry> modReg,
                      std::shared_ptr<ActivityRegistry> actReg,
                      ExceptionToActionTable const& actions);

    AllWorkers const& allWorkers() const { return allWorkers_; }

    void addToAllWorkers(Worker* w);

    ExceptionToActionTable const& actionTable() const { return *actionTable_; }

    template <typename T>
      requires requires(T const& x) { x.moduleDescription(); }
    Worker* getWorkerForModule(T const& module) {
      auto* worker = getWorkerForExistingModule(module.moduleDescription().moduleLabel());
      assert(worker != nullptr);
      assert(worker->matchesBaseClassPointer(static_cast<typename T::ModuleType const*>(&module)));
      return worker;
    }

    Worker* getWorkerForModule(edm::ModuleDescription const& iDescription) {
      auto* worker = getWorkerForExistingModule(iDescription.moduleLabel());
      assert(worker != nullptr);
      assert(worker->description() == &iDescription);
      return worker;
    }

    void resetAll();

  protected:
    Worker const* deleteModuleIfExists(std::string const& moduleLabel);
    AllWorkers& allWorkers() { return allWorkers_; }

    Worker* getWorkerForExistingModuleUnattached(std::string const& label) {
      return workerReg_.getWorkerFromExistingModule(label, actionTable_);
    }

    void setupResolvers(Principal& principal, UnscheduledAuxiliary const* aux);

  private:
    Worker* getWorkerForExistingModule(std::string const& label);

    WorkerRegistry<TI, TP> workerReg_;
    ExceptionToActionTable const* actionTable_;
    AllWorkers allWorkers_;
    void const* lastSetupPrincipal_;
  };

  template <typename TI, typename TP>
  class WorkerManager;
}  // namespace edm

#endif
