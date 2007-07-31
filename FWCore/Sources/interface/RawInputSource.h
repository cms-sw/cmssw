#ifndef FWCore_Sources_RawInputSource_h
#define FWCore_Sources_RawInputSource_h

/*----------------------------------------------------------------------
$Id: RawInputSource.h,v 1.4 2007/07/30 04:20:02 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/Timestamp.h"
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
    virtual std::auto_ptr<EventPrincipal> readEvent_(boost::shared_ptr<LuminosityBlockPrincipal>);
    virtual boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp);
    virtual boost::shared_ptr<RunPrincipal> readRun_();
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& eventID);
    virtual void skip(int offset);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    virtual void setRun(RunNumber_t r);
    
    int remainingEvents_;
    RunNumber_t runNumber_;
    LuminosityBlockNumber_t luminosityBlockNumber_;
    bool newRun_;
    bool newLumi_;
    std::auto_ptr<EventPrincipal> ep_;
    boost::shared_ptr<LuminosityBlockPrincipal> lbp_;
  };
}
#endif
