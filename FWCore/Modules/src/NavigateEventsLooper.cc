/** \class edm::NavigateEventsLooper

Allows interactive navigation from event to event
in cmsRun jobs.  Just add the looper to the
python configuration and then respond to the
questions that show up on the display by typing
a number followed by return.

This was originally written to test the looper interface
used by the Fireworks event display.  It might be useful
by itself.

If you use this either do not use a PoolOutputModule or
turn off fast cloning its configuration.

\author W. David Dagenhart, created 27 August, 2010

*/

#include "FWCore/Framework/interface/EDLooperBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ProcessingController.h"

#include <iostream>

namespace edm {

  class NavigateEventsLooper : public EDLooperBase {

  public:
    NavigateEventsLooper(ParameterSet const& pset);
    virtual ~NavigateEventsLooper();

    virtual void startingNewLoop(unsigned int iIteration);
    virtual Status duringLoop(Event const& ev, EventSetup const& es, ProcessingController& pc);
    virtual Status endOfLoop(EventSetup const& es, unsigned int iCounter);

  private:
    NavigateEventsLooper(NavigateEventsLooper const&); // stop default
    NavigateEventsLooper const& operator=(NavigateEventsLooper const&); // stop default

    int maxLoops_;
    int countLoops_;
    bool shouldStopLoop_;
    bool shouldStopProcess_;
  };

  NavigateEventsLooper::NavigateEventsLooper(ParameterSet const& pset) :
    maxLoops_(pset.getUntrackedParameter<int>("maxLoops", -1)),
    countLoops_(0),
    shouldStopLoop_(false),
    shouldStopProcess_(false) {
  }

  NavigateEventsLooper::~NavigateEventsLooper() {
  }
  
  void 
  NavigateEventsLooper::startingNewLoop(unsigned int iIteration) {
  }
  
  EDLooperBase::Status 
  NavigateEventsLooper::duringLoop(Event const& ev, EventSetup const& es, ProcessingController& pc) {

    if (!pc.lastOperationSucceeded()) {
      std::cout << "Event could not be found. Nothing done. Try again.\n";
    }

    std::cout << "\nWhat should we do next?\n";

    if (pc.forwardState() == ProcessingController::kEventsAheadInFile) {
      std::cout << "(0) process the next event\n";
    }
    else if (pc.forwardState() == ProcessingController::kNextFileExists) {
      std::cout << "(0) process the next event if it exists (at last event in the open file. there are more files)\n";
    }
    else if (pc.forwardState() == ProcessingController::kAtLastEvent) {
      std::cout << "(0) will stop the loop because this is the last event\n";
    }
    else if (pc.forwardState() == ProcessingController::kUnknownForward) {
      std::cout << "(0) process the next event (if it exists)\n";
    }

    if (pc.canRandomAccess()) {
      if (pc.reverseState() == ProcessingController::kEventsBackwardsInFile) {
        std::cout << "(1) process the previous event\n";
      }
      else if (pc.reverseState() == ProcessingController::kPreviousFileExists) {
        std::cout << "(1) process the previous event if there are any (at first event in the open file. there are previous files)\n";
      }
      else if (pc.reverseState() == ProcessingController::kAtFirstEvent) {
        std::cout << "(1) will stop the loop because this is the first event\n";
      }

      std::cout << "(2) process a specific event\n";
    }

    std::cout << "(3) stop loop\n";
    std::cout << "(4) stop process" << std::endl;
    int x;

    bool inputFailed = false;
    do {
      inputFailed = false;
      if (!(std::cin >> x) || x < 0 || x > 4) {
        inputFailed = true;
	std::cin.clear();
	std::cin.ignore(10000,'\n');
	std::cout << "Please enter numeric characters only. The value must be in the range 0 to 4 (inclusive). Please try again." << std::endl;
      }      
      if (!pc.canRandomAccess() && (x == 1 || x == 2)) {
        inputFailed = true;
	std::cout << "The source cannot do random access. 1 and 2 are illegal values. Please try again." << std::endl;        
      }
    } while (inputFailed);

    shouldStopLoop_ = false;
    shouldStopProcess_ = false;
    if (x == 0) {
      pc.setTransitionToNextEvent();
    }
    else if (x == 1) {
      pc.setTransitionToPreviousEvent();
    } 
    else if (x == 2) {
      std::cout << "Which run?" << std::endl;
      do {
        inputFailed = false;
        if (!(std::cin >> x)) {
          inputFailed = true;
	  std::cin.clear();
	  std::cin.ignore(10000,'\n');
	  std::cout << "Please enter numeric characters only. Please try again." << std::endl;
        }      
      } while (inputFailed);
      RunNumber_t run = x;
      std::cout << "Which luminosity block?" << std::endl;
      do {
        inputFailed = false;
        if (!(std::cin >> x)) {
          inputFailed = true;
	  std::cin.clear();
	  std::cin.ignore(10000,'\n');
	  std::cout << "Please enter numeric characters only. Please try again." << std::endl;
        }      
      } while (inputFailed);
      LuminosityBlockNumber_t lumi = x;
      std::cout << "Which event?" << std::endl;
      do {
        inputFailed = false;
        if (!(std::cin >> x)) {
          inputFailed = true;
	  std::cin.clear();
	  std::cin.ignore(10000,'\n');
	  std::cout << "Please enter numeric characters only. Please try again." << std::endl;
        }      
      } while (inputFailed);
      EventNumber_t ev = x;
      pc.setTransitionToEvent(EventID(run, lumi, ev));
    }
    else if (x == 3) {
      pc.setTransitionToNextEvent();
      shouldStopLoop_ = true;
    }
    else if (x == 4) {
      pc.setTransitionToNextEvent();
      shouldStopLoop_ = true;
      shouldStopProcess_ = true;
    }
    return shouldStopLoop_ ? kStop : kContinue;
  }
  
  EDLooperBase::Status 
  NavigateEventsLooper::endOfLoop(EventSetup const& es, unsigned int iCounter) {
    std::cout << "Ending loop" << std::endl;
    if (shouldStopProcess_) return kStop;
    ++countLoops_;
    return (maxLoops_ < 0 || countLoops_ < maxLoops_) ? kContinue : kStop;
  }
}

using edm::NavigateEventsLooper;
DEFINE_FWK_LOOPER(NavigateEventsLooper);
