#ifndef FWCore_Framework_UnscheduledCallProducer_h
#define FWCore_Framework_UnscheduledCallProducer_h

#include "FWCore/Framework/interface/UnscheduledHandler.h"

#include "FWCore/Framework/interface/BranchActionType.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>

namespace edm {

  class ModuleCallingContext;

  class UnscheduledCallProducer : public UnscheduledHandler {
  public:
    
    class WorkerLookup {
      //Compact way to quickly find workers or to iterate through all of them
    public:
      WorkerLookup() = default;
      
      using worker_container = std::vector<Worker*>;
      using const_iterator = worker_container::const_iterator;

      void add(Worker* iWorker) {
        auto const& label = iWorker->description().moduleLabel();
        size_t index = m_values.size();
        m_values.push_back(iWorker);
        if( not m_keys.emplace(label.c_str(),index).second) {
        //make sure keys are unique
          throw cms::Exception("WorkersWithSameLabel")<<"multiple workers use the label "<<label;
        }
      }
      
      Worker* find(std::string const& iLabel) const {
        auto found = m_keys.find(iLabel);
        if(found == m_keys.end()) {
          return nullptr;
        }
        return m_values[found->second];
      }
      
      const_iterator begin() const { return m_values.begin(); }
      const_iterator end() const { return m_values.end(); }
      
    private:
      //second element is the index of the key in m_values
      std::unordered_map<std::string, size_t> m_keys;
      worker_container m_values;
    };
    
    UnscheduledCallProducer() : UnscheduledHandler(), workerLookup_() {}
    void addWorker(Worker* aWorker) {
      assert(0 != aWorker);
      workerLookup_.add(aWorker);
    }

    template <typename T, typename U>
    void runNow(typename T::MyPrincipal& p, EventSetup const& es, StreamID streamID,
                typename T::Context const* topContext, U const* context) const {
      //do nothing for event since we will run when requested
      if(!T::isEvent_) {
        for(auto worker: workerLookup_) {
          try {
            ParentContext parentContext(context);
            worker->doWork<T>(p, es, streamID, parentContext, topContext);
          }
          catch (cms::Exception & ex) {
            std::ostringstream ost;
            if (T::isEvent_) {
              ost << "Calling event method";
            }
            else if (T::begin_ && T::branchType_ == InRun) {
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

  private:
    virtual bool tryToFillImpl(std::string const& moduleLabel,
                               EventPrincipal const& event,
                               EventSetup const& eventSetup,
                               ModuleCallingContext const* mcc) const override {
      auto worker =
        workerLookup_.find(moduleLabel);
      if(worker != nullptr) {
        try {
          ParentContext parentContext(mcc);
          worker->doWork<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> >(event,
              eventSetup, event.streamID(), parentContext, mcc->getStreamContext());
        }
        catch (cms::Exception & ex) {
          std::ostringstream ost;
          ost << "Calling produce method for unscheduled module " 
              << worker->description().moduleName() << "/'"
              << worker->description().moduleLabel() << "'";
          ex.addContext(ost.str());
          throw;
        }
        return true;
      }
      return false;
    }
    WorkerLookup workerLookup_;
  };

}

#endif

