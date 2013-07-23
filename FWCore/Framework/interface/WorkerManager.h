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
  
  class WorkerManager {
  public:
    typedef std::vector<Worker*> AllWorkers;

    WorkerManager(boost::shared_ptr<ActivityRegistry> actReg, ActionTable const& actions);

    void addToUnscheduledWorkers(ParameterSet& pset,
                      ProductRegistry& preg,
                      boost::shared_ptr<ProcessConfiguration> processConfiguration,
                      std::string label,
                      bool useStopwatch,
                      std::set<std::string>& unscheduledLabels,
                      std::vector<std::string>& shouldBeUsedLabels);

    void setOnDemandProducts(ProductRegistry& pregistry, std::set<std::string> const& unscheduledLabels) const;

    template <typename T>
    void processOneOccurrence(typename T::MyPrincipal& principal,
                              EventSetup const& eventSetup,
                              StreamID streamID,
                              bool cleaningUpAfterException = false);

    void beginJob(ProductRegistry const& iRegistry);
    void endJob();
    void endJob(ExceptionCollector& collector);

    void beginStream(StreamID iID);
    void endStream(StreamID iID);
    
    AllWorkers const& allWorkers() const {return allWorkers_;}

    void addToAllWorkers(Worker* w, bool useStopwatch);

    ActionTable const&  actionTable() const {return *actionTable_;}

    Worker* getWorker(ParameterSet& pset,
                      ProductRegistry& preg,
                      boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                      std::string label);

  private:

    void resetAll();

    void setupOnDemandSystem(EventPrincipal& principal, EventSetup const& es);

    WorkerRegistry      workerReg_;
    ActionTable const*  actionTable_;

    AllWorkers          allWorkers_;

    boost::shared_ptr<UnscheduledCallProducer> unscheduled_;
  };

  template <typename T>
  void
  WorkerManager::processOneOccurrence(typename T::MyPrincipal& ep,
                                 EventSetup const& es,
                                 StreamID streamID,
                                 bool cleaningUpAfterException) {
    this->resetAll();

    try {
      try {
        try {
          if (T::isEvent_) {
            setupOnDemandSystem(dynamic_cast<EventPrincipal&>(ep), es);
          } else {
            //make sure the unscheduled items see this run or lumi rtansition
            unscheduled_->runNow<T>(ep, es,streamID);
          }
        }
        catch(cms::Exception& e) {
          actions::ActionCodes action = (T::isEvent_ ? actionTable_->find(e.category()) : actions::Rethrow);
          assert (action != actions::IgnoreCompletely);
          assert (action != actions::FailPath);
          if (action == actions::SkipEvent) {
            printCmsExceptionWarning("SkipEvent", e);
          } else {
            throw;
          }
        }
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
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
