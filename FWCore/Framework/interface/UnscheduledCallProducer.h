#ifndef FWCore_Framework_UnscheduledCallProducer_h
#define FWCore_Framework_UnscheduledCallProducer_h

#include "FWCore/Framework/interface/UnscheduledHandler.h"

#include "FWCore/Framework/interface/BranchActionType.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/src/Worker.h"

#include <map>
#include <string>
#include <sstream>

namespace edm {
  class UnscheduledCallProducer : public UnscheduledHandler {
  public:
    UnscheduledCallProducer() : UnscheduledHandler(), labelToWorkers_() {}
    void addWorker(Worker* aWorker) {
      assert(0 != aWorker);
      labelToWorkers_[aWorker->description().moduleLabel()] = aWorker;
    }

    template <typename T>
    void runNow(typename T::MyPrincipal& p, EventSetup const& es, StreamID streamID) {
      //do nothing for event since we will run when requested
      if(!T::isEvent_) {
        for(std::map<std::string, Worker*>::iterator it = labelToWorkers_.begin(), itEnd=labelToWorkers_.end();
            it != itEnd;
            ++it) {
          CPUTimer timer;
          try {
            it->second->doWork<T>(p, es, nullptr, &timer,streamID);
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
            ost << " for unscheduled module " << it->second->description().moduleName()
                << "/'" << it->second->description().moduleLabel() << "'";
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
                               EventPrincipal& event,
                               EventSetup const& eventSetup,
                               CurrentProcessingContext const* iContext) {
      std::map<std::string, Worker*>::const_iterator itFound =
        labelToWorkers_.find(moduleLabel);
      if(itFound != labelToWorkers_.end()) {
        CPUTimer timer;
        try {
          itFound->second->doWork<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> >(event, eventSetup, iContext, &timer,event.streamID());
        }
        catch (cms::Exception & ex) {
	  std::ostringstream ost;
          ost << "Calling produce method for unscheduled module " 
              <<  itFound->second->description().moduleName() << "/'"
              << itFound->second->description().moduleLabel() << "'";
          ex.addContext(ost.str());
          throw;
        }
        return true;
      }
      return false;
    }
    std::map<std::string, Worker*> labelToWorkers_;
  };

}

#endif
