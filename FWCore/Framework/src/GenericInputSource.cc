/*----------------------------------------------------------------------
$Id: GenericInputSource.cc,v 1.1 2005/12/28 00:49:48 wmtan Exp $
----------------------------------------------------------------------*/

#include <stdexcept>
#include <memory>
#include <string>


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/GenericInputSource.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/EDProduct/interface/EventID.h"

namespace edm {
  //used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;
  
  GenericInputSource::GenericInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(desc),
    ProductRegistryHelper(),
    remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents_)),
    presentTime_(pset.getUntrackedParameter<unsigned int>("firstTime", 0)),  //time in ns
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec)),
    numberEventsInThisRun_(0),
    eventID_(pset.getUntrackedParameter<unsigned int>("firstRun", 1), 0),
    module_()
  { }

  GenericInputSource::~GenericInputSource() {
  }

  std::auto_ptr<EventPrincipal>
  GenericInputSource::read() {
    std::auto_ptr<EventPrincipal> result(0);
    
    if (remainingEvents_ != 0) {
      setRunAndEventInfo();
      result = std::auto_ptr<EventPrincipal>(new EventPrincipal(eventID_, Timestamp(presentTime_), productRegistry()));
      Event e(*result, module_);
      produce(e);
      e.commit_();
      ++numberEventsInThisRun_;
      --remainingEvents_;
    }
    return result;
  }
 
  void
  GenericInputSource::addToReg(ModuleDescription const& md) {
    module_ = md;
    if (!typeLabelList().empty()) {
      ProductRegistryHelper::addToRegistry(typeLabelList().begin(), typeLabelList().end(), md, productRegistry());
    }
  }

  void
  GenericInputSource::setRunAndEventInfo() {
    if (numberEventsInThisRun_ < numberEventsInRun_) {
      eventID_ = eventID_.next();
    } else {
      eventID_ = eventID_.nextRunFirstEvent();
      numberEventsInThisRun_ = 0;
    }
    presentTime_ += timeBetweenEvents_;
  }
}
