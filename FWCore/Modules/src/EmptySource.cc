/*----------------------------------------------------------------------
$Id: EmptySource.cc,v 1.1 2005/10/17 19:22:41 wmtan Exp $
----------------------------------------------------------------------*/

#include <stdexcept>
#include <memory>
#include <string>


#include "FWCore/Modules/src/EmptySource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/EDProduct/interface/EventID.h"

namespace edm {
  class BranchKey;

  //used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;
  
  EmptySource::EmptySource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    RandomAccessInputSource(desc),
    remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", 0)),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents_)),
    presentRun_(pset.getUntrackedParameter<unsigned int>("firstRun",1)),
    nextTime_(pset.getUntrackedParameter<unsigned int>("firstTime",1)),  //time in ns
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents",kNanoSecPerSec/kAveEventPerSec)),
    numberEventsInThisRun_(0),
    nextID_(presentRun_, 1)   
  { }

  EmptySource::~EmptySource() {
  }

  std::auto_ptr<EventPrincipal>
  EmptySource::read() {
    std::auto_ptr<EventPrincipal> result(0);
    
    if (remainingEvents_-- > 0) {
      result = std::auto_ptr<EventPrincipal>(new EventPrincipal(nextID_, Timestamp(nextTime_), *preg_));
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

  std::auto_ptr<EventPrincipal>
  EmptySource::read(EventID const& id) {
    nextID_ = id;
    presentRun_ = nextID_.run();
    return read();
  }

  void
  EmptySource::skip(int offset) {
    for(; offset < 0; ++offset) {
      nextID_ = nextID_.previous();
    }
    for(; offset > 0; --offset) {
      nextID_ = nextID_.next();
    }
    presentRun_ = nextID_.run();
  }
}
