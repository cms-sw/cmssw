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

#include "boost/shared_ptr.hpp"

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

    WorkerManager(boost::shared_ptr<ActivityRegistry> actReg, ExceptionToActionTable const& actions);

    WorkerManager(boost::shared_ptr<ModuleRegistry> modReg,
                  boost::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions);
    void addToUnscheduledWorkers(ParameterSet& pset,
                                 ProductRegistry& preg,
                                 PreallocationConfiguration const* prealloc,
                                 boost::shared_ptr<ProcessConfiguration> processConfiguration,
                                 std::string label,
                                 bool useStopwatch,
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

    void beginJob(ProductRegistry const& iRegistry);
    void endJob();
    void endJob(ExceptionCollector& collector);

    void beginStream(StreamID iID, StreamContext& streamContext);
    void endStream(StreamID iID, StreamContext& streamContext);
    
    AllWorkers const& allWorkers() const {return allWorkers_;}

    void addToAllWorkers(Worker* w, bool useStopwatch);

    ExceptionToActionTable const&  actionTable() const {return *actionTable_;}

    Worker* getWorker(ParameterSet& pset,
                      ProductRegistry& preg,
                      PreallocationConfiguration const* prealloc,
                      boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                      std::string const& label);

  private:

    void resetAll();

    void setupOnDemandSystem(EventPrincipal& principal, EventSetup const& es);

    WorkerRegistry      workerReg_;
    ExceptionToActionTable const*  actionTable_;

    AllWorkers          allWorkers_;

    boost::shared_ptr<UnscheduledCallProducer> unscheduled_;
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
        try {
          if (T::isEvent_) {
            setupOnDemandSystem(dynamic_cast<EventPrincipal&>(ep), es);
          } else {
            //make sure the unscheduled items see this run or lumi rtansition
            unscheduled_->runNow<T,U>(ep, es,streamID, topContext, context);
          }
        }
        catch(cms::Exception& e) {
          exception_actions::ActionCodes action = (T::isEvent_ ? actionTable_->find(e.category()) : exception_actions::Rethrow);
          assert (action != exception_actions::IgnoreCompletely);
          assert (action != exception_actions::FailPath);
          if (action == exception_actions::SkipEvent) {
            printCmsExceptionWarning("SkipEvent", e);
          } else {
            throw;
          }
        }
      });
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
