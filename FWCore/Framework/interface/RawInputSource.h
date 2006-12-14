#ifndef Framework_RawInputSource_h
#define Framework_RawInputSource_h

/*----------------------------------------------------------------------
$Id: RawInputSource.h,v 1.5 2006/12/01 03:29:52 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

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
    virtual void setRun(RunNumber_t r);
    
    int remainingEvents_;
    RunNumber_t runNumber_;
    RunNumber_t oldRunNumber_;
    std::auto_ptr<EventPrincipal> ep_;
    boost::shared_ptr<LuminosityBlockPrincipal const> luminosityBlockPrincipal_;
  };
}
#endif
