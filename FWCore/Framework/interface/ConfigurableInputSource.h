#ifndef Framework_ConfigurableInputSource_h
#define Framework_ConfigurableInputSource_h

/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.h,v 1.5 2006/05/31 22:02:57 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/InputSource.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

namespace edm {
  class Event;
  class InputSourceDescription;
  class ParameterSet;
  class ConfigurableInputSource : public InputSource {
  public:
    explicit ConfigurableInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~ConfigurableInputSource();

    unsigned long numberEventsInRun() const {return numberEventsInRun_;} 
    TimeValue_t presentTime() const {return presentTime_;}
    unsigned long timeBetweenEvents() const {return timeBetweenEvents_;}
    unsigned long numberEventsInThisRun() const {return numberEventsInThisRun_;}
    RunNumber_t run() const {return eventID_.run();}
    EventNumber_t event() const {return eventID_.event();}

  protected:

    void setEventNumber(EventNumber_t e) {
      RunNumber_t r = run();
      eventID_ = EventID(r, e);
    } 
    void setTime(TimeValue_t t) {presentTime_ = t;}

  private:
    virtual void setRunAndEventInfo();
    virtual bool produce(Event & e) = 0;
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& eventID);
    virtual std::auto_ptr<EventPrincipal> read();
    virtual void skip(int offset);
    virtual void setRun(RunNumber_t r) {eventID_ = EventID(r, zerothEvent_);} 
    virtual void rewind_() {
       eventID_ = origEventID_;
       presentTime_=origTime_;
    }
    
    unsigned long numberEventsInRun_;
    TimeValue_t presentTime_, origTime_;
    unsigned long timeBetweenEvents_;

    unsigned long numberEventsInThisRun_;
    unsigned long const zerothEvent_;
    EventID eventID_, origEventID_;
  };
}
#endif
