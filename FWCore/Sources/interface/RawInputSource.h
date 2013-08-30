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
    void makeEvent(EventPrincipal& eventPrincipal, EventAuxiliary const& eventAuxiliary);
    virtual bool checkNextEvent() = 0;
    virtual void read(EventPrincipal& eventPrincipal) = 0;
    void setInputFileTransitionsEachEvent() {inputFileTransitionsEachEvent_ = true;}

  private:
    virtual void readEvent_(EventPrincipal& eventPrincipal) override;
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_() override;
    virtual void reset_();
    virtual void rewind_() override;
    virtual ItemType getNextItemType() override;
    virtual void preForkReleaseResources() override;

    bool inputFileTransitionsEachEvent_;
  };
}
#endif
