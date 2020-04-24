#ifndef FWCore_Sources_RawInputSource_h
#define FWCore_Sources_RawInputSource_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <memory>
#include <utility>

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ParameterSet;
  class Timestamp;
  class RawInputSource : public InputSource {
  public:
    explicit RawInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    ~RawInputSource() override;
    static void fillDescription(ParameterSetDescription& description);

  protected:
    void makeEvent(EventPrincipal& eventPrincipal, EventAuxiliary const& eventAuxiliary);
    virtual bool checkNextEvent() = 0;
    virtual void read(EventPrincipal& eventPrincipal) = 0;
    void setInputFileTransitionsEachEvent() {inputFileTransitionsEachEvent_ = true;}

  private:
    void readEvent_(EventPrincipal& eventPrincipal) override;
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_() override;
    virtual void reset_();
    void rewind_() override;
    ItemType getNextItemType() override;
    void closeFile_() final;
    virtual void genuineCloseFile() { }

    bool inputFileTransitionsEachEvent_;
    bool fakeInputFileTransition_;
  };
}
#endif
