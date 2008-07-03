#ifndef FWCore_Sources_RawInputSource_h
#define FWCore_Sources_RawInputSource_h

/*----------------------------------------------------------------------
$Id: RawInputSource.h,v 1.9 2007/12/11 00:28:07 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ParameterSet;
  class Timestamp;
  class RawInputSource : public InputSource {
  public:
    explicit RawInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~RawInputSource();

  protected:
    std::auto_ptr<Event> makeEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, Timestamp const& tstamp);
    virtual std::auto_ptr<Event> readOneEvent() = 0;

  private:
    virtual std::auto_ptr<EventPrincipal> readEvent_(boost::shared_ptr<LuminosityBlockPrincipal>);
    virtual boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_();
    virtual boost::shared_ptr<RunPrincipal> readRun_();
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const& eventID);
    virtual void skip(int offset);
    virtual InputSource::ItemType getNextItemType();
    
    RunNumber_t runNumber_;
    LuminosityBlockNumber_t luminosityBlockNumber_;
    bool newRun_;
    bool newLumi_;
    std::auto_ptr<EventPrincipal> ep_;
  };
}
#endif
