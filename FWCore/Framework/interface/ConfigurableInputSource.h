#ifndef Framework_ConfigurableInputSource_h
#define Framework_ConfigurableInputSource_h

/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.h,v 1.12 2006/12/21 00:05:35 wmtan Exp $
----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/LuminosityBlockID.h"
#include "DataFormats/Common/interface/RunID.h"

namespace edm {
  class ParameterSet;
  class ConfigurableInputSource : public InputSource {
  public:
    explicit ConfigurableInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~ConfigurableInputSource();

    unsigned int numberEventsInRun() const {return numberEventsInRun_;} 
    unsigned int numberEventsInLumi() const {return numberEventsInLumi_;} 
    TimeValue_t presentTime() const {return presentTime_;}
    unsigned int timeBetweenEvents() const {return timeBetweenEvents_;}
    unsigned int numberEventsInThisRun() const {return numberEventsInThisRun_;}
    unsigned int numberEventsInThisLumi() const {return numberEventsInThisLumi_;}
    RunNumber_t run() const {return eventID_.run();}
    EventNumber_t event() const {return eventID_.event();}
    LuminosityBlockID luminosityBlockID() const {return luminosityBlockID_;}

  protected:

    void setEventNumber(EventNumber_t e) {
      RunNumber_t r = run();
      eventID_ = EventID(r, e);
    } 
    void setTime(TimeValue_t t) {presentTime_ = t;}

  private:
    void finishRun();
    void finishLumi();
    virtual void setRunAndEventInfo();
    virtual bool produce(Event & e) = 0;
    virtual void beginRun(Run &) {}
    virtual void endRun(Run &) {}
    virtual void beginLuminosityBlock(LuminosityBlock &) {}
    virtual void endLuminosityBlock(LuminosityBlock &) {}
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& eventID);
    virtual std::auto_ptr<EventPrincipal> read();
    virtual void skip(int offset);
    virtual void setRun(RunNumber_t r);
    virtual void setLumi(LuminosityBlockID lb);
    virtual void rewind_();
    
    unsigned int numberEventsInRun_;
    unsigned int numberEventsInLumi_;
    TimeValue_t presentTime_;
    TimeValue_t origTime_;
    unsigned int timeBetweenEvents_;

    unsigned int numberEventsInThisRun_;
    unsigned int numberEventsInThisLumi_;
    unsigned int const zerothEvent_;
    EventID eventID_;
    EventID origEventID_;
    LuminosityBlockID luminosityBlockID_;
    LuminosityBlockID origLuminosityBlockID_;
    bool justBegun_;

    boost::shared_ptr<LuminosityBlockPrincipal const> luminosityBlockPrincipal_;
  };
}
#endif
