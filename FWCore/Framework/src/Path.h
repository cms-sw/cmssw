#ifndef FWCore_Framework_Path_h
#define FWCore_Framework_Path_h

/*
  Author: Jim Kowalkowski 28-01-06

  An object of this type represents one path in a job configuration.
  It holds the assigned bit position and the list of workers that are
  an event must pass through when this parh is processed.  The workers
  are held in WorkerInPath wrappers so that per path execution statistics
  can be kept for each worker.
*/

#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/Worker.h"
#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>
#include <map>
#include <exception>
#include <sstream>

#include "FWCore/Framework/src/RunStopwatch.h"

namespace edm {
  class EventPrincipal;
  class RunPrincipal;
  class LuminosityBlockPrincipal;
  class EarlyDeleteHelper;
  class StreamID;

  class Path {
  public:
    typedef hlt::HLTState State;

    typedef std::vector<WorkerInPath> WorkersInPath;
    typedef WorkersInPath::size_type        size_type;
    typedef boost::shared_ptr<HLTGlobalStatus> TrigResPtr;

    Path(int bitpos, std::string const& path_name,
         WorkersInPath const& workers,
         TrigResPtr trptr,
         ExceptionToActionTable const& actions,
         boost::shared_ptr<ActivityRegistry> reg,
         StreamContext const* streamContext,
         PathContext::PathType pathType);

    Path(Path const&);

    template <typename T>
    void processOneOccurrence(typename T::MyPrincipal&, EventSetup const&,
                              StreamID const&, typename T::Context const*);

    int bitPosition() const { return bitpos_; }
    std::string const& name() const { return name_; }

    std::pair<double, double> timeCpuReal() const {
      if(stopwatch_) {
        return std::pair<double, double>(stopwatch_->cpuTime(), stopwatch_->realTime());
      }
      return std::pair<double, double>(0., 0.);
    }

    std::pair<double, double> timeCpuReal(unsigned int const i) const {
      return workers_.at(i).timeCpuReal();
    }

    void clearCounters();

    int timesRun() const { return timesRun_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }
    //int abortWorker() const { return abortWorker_; }
    State state() const { return state_; }

    size_type size() const { return workers_.size(); }
    int timesVisited(size_type i) const { return workers_.at(i).timesVisited(); }
    int timesPassed (size_type i) const { return workers_.at(i).timesPassed() ; }
    int timesFailed (size_type i) const { return workers_.at(i).timesFailed() ; }
    int timesExcept (size_type i) const { return workers_.at(i).timesExcept() ; }
    Worker const* getWorker(size_type i) const { return workers_.at(i).getWorker(); }
    
    void setEarlyDeleteHelpers(std::map<const Worker*,EarlyDeleteHelper*> const&);

    void useStopwatch();
  private:

    // If you define this be careful about the pointer in the
    // PlaceInPathContext object in the contained WorkerInPath objects.
    Path const& operator=(Path const&) = delete; // stop default

    RunStopwatch::StopwatchPointer stopwatch_;
    int timesRun_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    //int abortWorker_;
    State state_;

    int bitpos_;
    std::string name_;
    TrigResPtr trptr_;
    boost::shared_ptr<ActivityRegistry> actReg_;
    ExceptionToActionTable const* act_table_;

    WorkersInPath workers_;
    std::vector<EarlyDeleteHelper*> earlyDeleteHelpers_;

    PathContext pathContext_;

    // Helper functions
    // nwrwue = numWorkersRunWithoutUnhandledException (really!)
    bool handleWorkerFailure(cms::Exception & e,
                             int nwrwue,
                             bool isEvent,
                             bool begin,
                             BranchType branchType,
                             CurrentProcessingContext const& cpc,
                             std::string const& id);
    static void exceptionContext(cms::Exception & ex,
                                 bool isEvent,
                                 bool begin,
                                 BranchType branchType,
                                 CurrentProcessingContext const& cpc,
                                 std::string const& id);
    void recordStatus(int nwrwue, bool isEvent);
    void updateCounters(bool succeed, bool isEvent);
    
    void handleEarlyFinish(EventPrincipal&);
    void handleEarlyFinish(RunPrincipal&) {}
    void handleEarlyFinish(LuminosityBlockPrincipal&) {}
  };

  namespace {
    template <typename T>
    class PathSignalSentry {
    public:
      PathSignalSentry(ActivityRegistry *a,
                       std::string const& name,
                       int const& nwrwue,
                       hlt::HLTState const& state,
                       PathContext const* pathContext) :
        a_(a), name_(name), nwrwue_(nwrwue), state_(state), pathContext_(pathContext) {
        if (a_) T::prePathSignal(a_, name_, pathContext_);
      }
      ~PathSignalSentry() {
        HLTPathStatus status(state_, nwrwue_);
        if(a_) T::postPathSignal(a_, name_, status, pathContext_);
      }
    private:
      ActivityRegistry* a_;
      std::string const& name_;
      int const& nwrwue_;
      hlt::HLTState const& state_;
      PathContext const* pathContext_;
    };
  }

  template <typename T>
  void Path::processOneOccurrence(typename T::MyPrincipal& ep, EventSetup const& es,
                                  StreamID const& streamID, typename T::Context const* context) {

    //Create the PathSignalSentry before the RunStopwatch so that
    // we only record the time spent in the path not from the signal
    int nwrwue = -1;
    PathSignalSentry<T> signaler(actReg_.get(), name_, nwrwue, state_, &pathContext_);

    // A RunStopwatch, but only if we are processing an event.
    RunStopwatch stopwatch(T::isEvent_ ? stopwatch_ : RunStopwatch::StopwatchPointer());

    if (T::isEvent_) {
      ++timesRun_;
    }
    state_ = hlt::Ready;

    // nwrue =  numWorkersRunWithoutUnhandledException
    bool should_continue = true;
    CurrentProcessingContext cpc(&name_, bitPosition(), pathContext_.pathType() == PathContext::PathType::kEndPath);

    WorkersInPath::size_type idx = 0;
    // It seems likely that 'nwrwue' and 'idx' can never differ ---
    // if so, we should remove one of them!.
    for (WorkersInPath::iterator i = workers_.begin(), end = workers_.end();
          i != end && should_continue;
          ++i, ++idx) {
      ++nwrwue;
      assert (static_cast<int>(idx) == nwrwue);
      try {
        try {
          cpc.activate(idx, i->getWorker()->descPtr());
          if(T::isEvent_) {
            should_continue = i->runWorker<T>(ep, es, &cpc, streamID, context);
          } else {
            should_continue = i->runWorker<T>(ep, es, &cpc, streamID, context);
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
        // handleWorkerFailure may throw a new exception.
	std::ostringstream ost;
        ost << ep.id();
        should_continue = handleWorkerFailure(ex, nwrwue, T::isEvent_, T::begin_, T::branchType_, cpc, ost.str());
      }
    }
    if (not should_continue) {
      handleEarlyFinish(ep);
    }
    updateCounters(should_continue, T::isEvent_);
    recordStatus(nwrwue, T::isEvent_);
  }

}

#endif
