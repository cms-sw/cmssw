#ifndef Framework_ConfigurableInputSource_h
#define Framework_ConfigurableInputSource_h

/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.h,v 1.1 2006/01/18 00:38:44 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GenericInputSource.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

namespace edm {
  class Event;
  class InputSourceDescription;
  class ParameterSet;
  class ConfigurableInputSource : public GenericInputSource {
  public:
    explicit ConfigurableInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~ConfigurableInputSource();

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
    virtual std::auto_ptr<EventPrincipal> read(EventID const& eventID);
    virtual void skip (int offset);
    
    int remainingEvents_;
    unsigned long numberEventsInRun_;
    TimeValue_t presentTime_;
    unsigned long timeBetweenEvents_;

    unsigned long numberEventsInThisRun_;
    EventID eventID_;
  };
}
#endif
