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
    void processOneOccurrence(typename T::MyPrincipal& principal,
                              EventSetup const& eventSetup,
                              StreamID streamID,
                              typename T::Context const* topContext,
                              U const* context,
                              bool cleaningUpAfterException = false);

    void setupOnDemandSystem(EventPrincipal& principal, EventSetup const& es);

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

  private:

    void resetAll();

    WorkerRegistry      workerReg_;
    ExceptionToActionTable const*  actionTable_;
    AllWorkers          allWorkers_;
    UnscheduledCallProducer unscheduled_;
    void const* lastSetupEventPrincipal_;
  };

  template <typename T, typename U>
  void
  WorkerManager::processOneOccurrence(typename T::MyPrincipal& ep,
                                 EventSetup const& es,
                                 StreamID streamID,
                                 typename T::Context const* topContext,
                                 U const* context,
                                 bool cleaningUpAfterException) {
    this->resetAll();

    try {
      convertException::wrap([&]() {
        //make sure the unscheduled items see this run or lumi transition
        unscheduled_.runNow<T,U>(ep, es,streamID, topContext, context);
        }
      );
    }
    catch(cms::Exception& ex) {
      if (ex.context().empty()) {
        addContextAndPrintException("Calling function WorkerManager::processOneOccurrence", ex, cleaningUpAfterException);
      } else {
        addContextAndPrintException("", ex, cleaningUpAfterException);
      }
      throw;
    }
  }
}

#endif
