#ifndef FWCore_Sources_RawInputSource_h
#define FWCore_Sources_RawInputSource_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <memory>

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
    virtual EventPrincipal* readEvent_();
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual EventPrincipal* readIt(EventID const& eventID);
    virtual void skip(int offset);
    virtual ItemType getNextItemType();
    
    RunNumber_t runNumber_;
    LuminosityBlockNumber_t luminosityBlockNumber_;
    bool newRun_;
    bool newLumi_;
    bool eventCached_;
  };
}
#endif
