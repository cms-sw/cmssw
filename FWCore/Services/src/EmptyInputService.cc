/*----------------------------------------------------------------------
$Id: EmptyInputService.cc,v 1.11 2005/08/10 15:28:12 chrjones Exp $
----------------------------------------------------------------------*/

#include <stdexcept>
#include <memory>
#include <string>

#include "FWCore/Services/src/EmptyInputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/InputServiceDescription.h"
#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/EDProduct/interface/EventID.h"

namespace edm {
  class BranchKey;
  FakeRetriever::~FakeRetriever() {}

  std::auto_ptr<EDProduct>
  FakeRetriever::get(BranchKey const&) {
    throw std::runtime_error("FakeRetriever::get called");
  }

  //used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;
  
  EmptyInputService::EmptyInputService(ParameterSet const& pset,
				       InputServiceDescription const& desc) :
    InputService(desc),
    remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
    retriever_(new FakeRetriever()),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents_+1)),
    presentRun_(pset.getUntrackedParameter<unsigned int>("firstRun",1)),
    nextTime_(pset.getUntrackedParameter<unsigned int>("firstTime",1)),  //time in ns
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents",kNanoSecPerSec/kAveEventPerSec)),
    numberEventsInThisRun_(0),
    nextID_(presentRun_, 1)   
  { }

  EmptyInputService::~EmptyInputService() {
    delete retriever_;
  }

  std::auto_ptr<EventPrincipal>
  EmptyInputService::read() {
    std::auto_ptr<EventPrincipal> result(0);
    
    if (remainingEvents_-- != 0) {
      result = std::auto_ptr<EventPrincipal>(new EventPrincipal(nextID_, Timestamp(nextTime_),*retriever_, *preg_));
      if(++numberEventsInThisRun_ < numberEventsInRun_) {
        nextID_ = nextID_.next();
      } else {
        nextID_ = nextID_.nextRunFirstEvent();
        numberEventsInThisRun_ = 0;
      }
      nextTime_ += timeBetweenEvents_;
    }
    return result;
  }
}
