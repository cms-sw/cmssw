#ifndef Framework_RawInputSource_h
#define Framework_RawInputSource_h

/*----------------------------------------------------------------------
$Id: RawInputSource.h,v 1.10 2007/03/04 06:00:22 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ParameterSet;
  class Timestamp;
  class RawInputSource : public InputSource {
  public:
    explicit RawInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~RawInputSource();

    int remainingEvents() const {return remainingEvents_;}

  protected:
    std::auto_ptr<Event> makeEvent(EventID & eventId, Timestamp const& tstamp);
    virtual std::auto_ptr<Event> readOneEvent() = 0;

  private:
    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& eventID);
    virtual void skip(int offset);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    virtual void setRun(RunNumber_t r);
    
    int remainingEvents_;
    RunNumber_t runNumber_;
    LuminosityBlockNumber_t luminosityBlockNumber_;

    std::auto_ptr<EventPrincipal> ep_;
  };
}
#endif
