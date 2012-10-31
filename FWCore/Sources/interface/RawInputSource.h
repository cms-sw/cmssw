#ifndef FWCore_Sources_RawInputSource_h
#define FWCore_Sources_RawInputSource_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <memory>
#include <utility>

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
    EventPrincipal* makeEvent(EventPrincipal& eventPrincipal, EventAuxiliary const& eventAuxiliary);
    virtual bool checkNextEvent() = 0;
    virtual EventPrincipal* read(EventPrincipal& eventPrincipal) = 0;
    void setInputFileTransitionsEachEvent() {inputFileTransitionsEachEvent_ = true;}

  private:
    virtual EventPrincipal* readEvent_(EventPrincipal& eventPrincipal);
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual void reset_();
    virtual void rewind_();
    virtual ItemType getNextItemType();
    virtual void preForkReleaseResources();
    virtual void postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>);

    bool inputFileTransitionsEachEvent_;

    //used when process has been forked
    boost::shared_ptr<edm::multicore::MessageReceiverForSource> receiver_;
    unsigned int numberOfEventsBeforeBigSkip_;
  };
}
#endif
