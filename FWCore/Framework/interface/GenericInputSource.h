#ifndef Framework_GenericInputSource_h
#define Framework_GenericInputSource_h

/*----------------------------------------------------------------------
$Id: GenericInputSource.h,v 1.2 2005/12/28 21:49:50 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/EDProduct/interface/Timestamp.h"

namespace edm {
  class Event;
  class InputSourceDescription;
  class ParameterSet;
  class GenericInputSource : public InputSource, public ProductRegistryHelper {
  public:
    explicit GenericInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~GenericInputSource();

    int remainingEvents() const {return remainingEvents_;}
    unsigned long numberEventsInRun() const {return numberEventsInRun_;} 
    TimeValue_t presentTime() const {return presentTime_;}
    unsigned long timeBetweenEvents() const {return timeBetweenEvents_;}
    unsigned long numberEventsInThisRun() const {return numberEventsInThisRun_;}
    RunNumber_t run() const {return eventID_.run();}
    EventNumber_t event() const {return eventID_.event();}

  protected:
    virtual void setRunAndEventInfo();
    virtual bool produce(Event & e) = 0;

    void setRunNumber(RunNumber_t r) {eventID_ = EventID(r, 0);} 
    void setEventNumber(EventNumber_t e) {
      RunNumber_t r = run();
      eventID_ = EventID(r, e);
    } 
    void setTime(TimeValue_t t) {presentTime_ = t;}

  private:
    virtual std::auto_ptr<EventPrincipal> read();
    
    virtual void addToReg(ModuleDescription const& md);

    int remainingEvents_;
    unsigned long numberEventsInRun_;
    TimeValue_t presentTime_;
    unsigned long timeBetweenEvents_;

    unsigned long numberEventsInThisRun_;
    EventID eventID_;
    ModuleDescription module_;
  };
}
#endif
