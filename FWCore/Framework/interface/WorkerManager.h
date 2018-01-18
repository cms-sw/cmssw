#ifndef FWCore_Framework_WorkerManager_h
#define FWCore_Framework_WorkerManager_h

/*

*/

#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/UnscheduledCallProducer.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <memory>

#include <set>
#include <string>
#include <vector>

namespace edm {
  class ExceptionCollector;
  class StreamID;
  class StreamContext;
  class ModuleRegistry;
  class PreallocationConfiguration;
  
  class WorkerManager {
  public:
    typedef std::vector<Worker*> AllWorkers;

    WorkerManager(std::shared_ptr<ActivityRegistry> actReg, ExceptionToActionTable const& actions);

    WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                  std::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions);
    void addToUnscheduledWorkers(ParameterSet& pset,
                                 ProductRegistry& preg,
                                 PreallocationConfiguration const* prealloc,
                                 std::shared_ptr<ProcessConfiguration> processConfiguration,
                                 std::string label,
                                 std::set<std::string>& unscheduledLabels,
                                 std::vector<std::string>& shouldBeUsedLabels);

    void setOnDemandProducts(ProductRegistry& pregistry, std::set<std::string> const& unscheduledLabels) const;

    template <typename T, typename U>
    void processOneOccurrenceAsync(
                              WaitingTask* task,
                              typename T::MyPrincipal& principal,
                              EventSetup const& eventSetup,
                              StreamID streamID,
                              typename T::Context const* topContext,
                              U const* context);

    template <typename T>
    void processAccumulatorsAsync(WaitingTask* task,
                                 typename T::MyPrincipal const& ep,
                                 EventSetup const& es,
                                 StreamID streamID,
                                 ParentContext const& parentContext,
                                 typename T::Context const* context);

    void setupOnDemandSystem(Principal& principal, EventSetup const& es);

    void beginJob(ProductRegistry const& iRegistry);
    void endJob();
    void endJob(ExceptionCollector& collector);

    void beginStream(StreamID iID, StreamContext& streamContext);
    void endStream(StreamID iID, StreamContext& streamContext);
    
    AllWorkers const& allWorkers() const {return allWorkers_;}

    void addToAllWorkers(Worker* w);

    ExceptionToActionTable const&  actionTable() const {return *actionTable_;}

    Worker* getWorker(ParameterSet& pset,
                      ProductRegistry& preg,
                      PreallocationConfiguration const* prealloc,
                      std::shared_ptr<ProcessConfiguration const> processConfiguration,
                      std::string const& label);

    void resetAll();

  private:

    WorkerRegistry      workerReg_;
    ExceptionToActionTable const*  actionTable_;
    AllWorkers          allWorkers_;
    UnscheduledCallProducer unscheduled_;
    void const* lastSetupEventPrincipal_;
  };

  template <typename T, typename U>
  void
  WorkerManager::processOneOccurrenceAsync(WaitingTask* task,
                                           typename T::MyPrincipal& ep,
                                           EventSetup const& es,
                                           StreamID streamID,
                                           typename T::Context const* topContext,
                                           U const* context) {
    //make sure the unscheduled items see this run or lumi transition
    unscheduled_.runNowAsync<T,U>(task,ep, es,streamID, topContext, context);
  }

  template <typename T>
  void
  WorkerManager::processAccumulatorsAsync(WaitingTask* task,
                                          typename T::MyPrincipal const& ep,
                                          EventSetup const& es,
                                          StreamID streamID,
                                          ParentContext const& parentContext,
                                          typename T::Context const* context) {
    unscheduled_.runAccumulatorsAsync<T>(task, ep, es, streamID, parentContext, context);
  }
}

#endif
