#ifndef FWCore_Sources_RawInputSource_h
#define FWCore_Sources_RawInputSource_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

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
    static void fillDescription(ParameterSetDescription& description);

  protected:
    EventPrincipal* makeEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, Timestamp const& tstamp);
    virtual EventPrincipal* read() = 0;
    void setInputFileTransitionsEachEvent() {inputFileTransitionsEachEvent_ = true;}

  private:
    virtual EventPrincipal* readEvent_();
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual EventPrincipal* readIt(EventID const& eventID);
    virtual void skip(int offset);
    virtual ItemType getNextItemType();
    bool inputFileTransitionsEachEvent_;
  };
}
#endif
