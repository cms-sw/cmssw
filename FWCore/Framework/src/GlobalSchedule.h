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

#include "boost/shared_ptr.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>

namespace edm {

  namespace {
    template <typename T>
    class GlobalScheduleSignalSentry {
    public:
      GlobalScheduleSignalSentry(ActivityRegistry* a, typename T::MyPrincipal* principal, EventSetup const* es, typename T::Context const* context) :
        a_(a), principal_(principal), es_(es), context_(context) {
        if (a_) T::preScheduleSignal(a_, principal_, context_);
      }
      ~GlobalScheduleSignalSentry() {
        if (a_) if (principal_) T::postScheduleSignal(a_, principal_, es_, context_);
      }

    private:
      // We own none of these resources.
      ActivityRegistry* a_;
      typename T::MyPrincipal* principal_;
      EventSetup const* es_;
      typename T::Context const* context_;
    };
  }

  class ActivityRegistry;
  class EventSetup;
  class ExceptionCollector;
  class ProcessContext;
  class PreallocationConfiguration;
  class ModuleRegistry;
  class TriggerResultInserter;
  
  class GlobalSchedule {
  public:
    typedef std::vector<std::string> vstring;
    typedef std::vector<Worker*> AllWorkers;
    typedef boost::shared_ptr<Worker> WorkerPtr;
    typedef std::vector<Worker*> Workers;

    GlobalSchedule(TriggerResultInserter* inserter,
                   boost::shared_ptr<ModuleRegistry> modReg,
                   std::vector<std::string> const& modulesToUse,
                   ParameterSet& proc_pset,
                   ProductRegistry& pregistry,
                   ExceptionToActionTable const& actions,
                   boost::shared_ptr<ActivityRegistry> areg,
                   boost::shared_ptr<ProcessConfiguration> processConfiguration,
                   ProcessContext const* processContext);
    GlobalSchedule(GlobalSchedule const&) = delete;

    template <typename T>
    void processOneGlobal(typename T::MyPrincipal& principal,
                          EventSetup const& eventSetup,
                          bool cleaningUpAfterException = false);

    void beginJob(ProductRegistry const&);
    void endJob(ExceptionCollector & collector);
    
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
    AllWorkers const& allWorkers() const {
      return workerManager_.allWorkers();
    }

  private:
    
    template<typename T>
    void runNow(typename T::MyPrincipal& p, EventSetup const& es,
                GlobalContext const* context);

    /// returns the action table
    ExceptionToActionTable const& actionTable() const {
      return workerManager_.actionTable();
    }
    
    void addToAllWorkers(Worker* w);
    
    WorkerManager                         workerManager_;
    boost::shared_ptr<ActivityRegistry>   actReg_;
    WorkerPtr                             results_inserter_;


    ProcessContext const*                 processContext_;
  };


  template <typename T>
  void
  GlobalSchedule::processOneGlobal(typename T::MyPrincipal& ep,
                                 EventSetup const& es,
                                 bool cleaningUpAfterException) {
    GlobalContext globalContext = T::makeGlobalContext(ep, processContext_);

    GlobalScheduleSignalSentry<T> sentry(actReg_.get(), &ep, &es, &globalContext);

    // This call takes care of the unscheduled processing.
    workerManager_.processOneOccurrence<T>(ep, es, StreamID::invalidStreamID(), &globalContext, &globalContext, cleaningUpAfterException);

    try {
      try {
        runNow<T>(ep,es,&globalContext);
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
        addContextAndPrintException("Calling function GlobalSchedule::processOneGlobal", ex, cleaningUpAfterException);
      } else {
        addContextAndPrintException("", ex, cleaningUpAfterException);
      }
      throw;
    }
  }
  template <typename T>
  void
  GlobalSchedule::runNow(typename T::MyPrincipal& p, EventSetup const& es,
              GlobalContext const* context) {
    //do nothing for event since we will run when requested
    for(auto & worker: allWorkers()) {
      try {
        ParentContext parentContext(context);
        worker->doWork<T>(p, es, nullptr,StreamID::invalidStreamID(), parentContext, context);
      }
      catch (cms::Exception & ex) {
        std::ostringstream ost;
        if (T::begin_ && T::branchType_ == InRun) {
          ost << "Calling beginRun";
        }
        else if (T::begin_ && T::branchType_ == InLumi) {
          ost << "Calling beginLuminosityBlock";
        }
        else if (!T::begin_ && T::branchType_ == InLumi) {
          ost << "Calling endLuminosityBlock";
        }
        else if (!T::begin_ && T::branchType_ == InRun) {
          ost << "Calling endRun";
        }
        else {
          // It should be impossible to get here ...
          ost << "Calling unknown function";
        }
        ost << " for unscheduled module " << worker->description().moduleName()
        << "/'" << worker->description().moduleLabel() << "'";
        ex.addContext(ost.str());
        ost.str("");
        ost << "Processing " << p.id();
        ex.addContext(ost.str());
        throw;
      }
    }
  }
}

#endif
