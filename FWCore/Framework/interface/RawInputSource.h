#ifndef Framework_RawInputSource_h
#define Framework_RawInputSource_h

/*----------------------------------------------------------------------
$Id: RawInputSource.h,v 1.1 2006/01/18 00:38:44 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/InputSource.h"

namespace edm {
  class Event;
  class EventPrincipal;
  class InputSourceDescription;
  class ParameterSet;
  class Timestamp;
  class RawInputSource : public InputSource {
  public:
    explicit RawInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~RawInputSource();

    int remainingEvents() const {return remainingEvents_;}

  protected:
    std::auto_ptr<Event> makeEvent(EventID const& eventId, Timestamp const& tstamp);
    virtual std::auto_ptr<Event> readOneEvent() = 0;

  private:
    virtual std::auto_ptr<EventPrincipal> read();
    
    int remainingEvents_;
    std::auto_ptr<EventPrincipal> ep_;
  };
}
#endif
