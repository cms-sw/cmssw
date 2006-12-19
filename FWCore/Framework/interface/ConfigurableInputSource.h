#ifndef Framework_ConfigurableInputSource_h
#define Framework_ConfigurableInputSource_h

/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.h,v 1.10 2006/12/14 04:30:57 wmtan Exp $
----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

namespace edm {
  class ParameterSet;
  class ConfigurableInputSource : public InputSource {
  public:
    explicit ConfigurableInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~ConfigurableInputSource();

    unsigned int numberEventsInRun() const {return numberEventsInRun_;} 
    TimeValue_t presentTime() const {return presentTime_;}
    unsigned int timeBetweenEvents() const {return timeBetweenEvents_;}
    unsigned int numberEventsInThisRun() const {return numberEventsInThisRun_;}
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
    virtual bool beginRun(Run &){return true;}
    virtual void endRun(Run &){}
    virtual bool beginLiminosityBlock(LuminosityBlock &){return true;}
    virtual void endLuminosityBlock(Run &){}
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& eventID);
    virtual std::auto_ptr<EventPrincipal> read();
    virtual void skip(int offset);
    virtual void setRun(RunNumber_t r) {eventID_ = EventID(r, zerothEvent_);} 
    virtual void rewind_() {
       eventID_ = origEventID_;
       presentTime_ = origTime_;
    }
    
    unsigned int numberEventsInRun_;
    TimeValue_t presentTime_;
    TimeValue_t origTime_;
    unsigned int timeBetweenEvents_;

    unsigned int numberEventsInThisRun_;
    unsigned int const zerothEvent_;
    EventID eventID_;
    EventID origEventID_;

    boost::shared_ptr<LuminosityBlockPrincipal const> luminosityBlockPrincipal_;
  };
}
#endif
