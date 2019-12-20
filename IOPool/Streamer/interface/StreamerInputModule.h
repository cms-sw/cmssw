#ifndef IOPool_Streamer_StreamerInputModule_h
#define IOPool_Streamer_StreamerInputModule_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include <iterator>

namespace edm {
  template <typename Producer>
  class StreamerInputModule : public StreamerInputSource {
    /**
     Requires the Producer class to provide following functions
           const InitMsgView* getHeader();
           const EventMsgView* getNextEvent();
           bool newHeader() const;
  */
  public:
    explicit StreamerInputModule(ParameterSet const& pset, InputSourceDescription const& desc);
    ~StreamerInputModule() override;

  private:
    void genuineCloseFile() override {
      if (pr_.get() != nullptr)
        pr_->closeFile();
    }

    void genuineReadFile() override {
      if (isFirstFile_) {
        isFirstFile_ = false;
        return;
      }

      InitMsgView const* header = pr_->getHeader();
      deserializeAndMergeWithRegistry(*header);
    }

    Next checkNext() override;

    edm::propagate_const<std::unique_ptr<Producer>> pr_;
    bool isFirstFile_ = true;
  };  //end-of-class-def

  template <typename Producer>
  StreamerInputModule<Producer>::~StreamerInputModule() {}

  template <typename Producer>
  StreamerInputModule<Producer>::StreamerInputModule(ParameterSet const& pset, InputSourceDescription const& desc)
      : StreamerInputSource(pset, desc),
        //prod_reg_(&productRegistry()),
        pr_(new Producer(pset)) {
    //Get header/init from Producer
    InitMsgView const* header = pr_->getHeader();
    deserializeAndMergeWithRegistry(*header);
  }

  template <typename Producer>
  StreamerInputSource::Next StreamerInputModule<Producer>::checkNext() {
    EventMsgView const* eview = pr_->getNextEvent();

    if (pr_->newHeader()) {
      FDEBUG(6) << "A new file has been opened and we must compare Headers here !!" << std::endl;
      return Next::kFile;
    }
    if (eview == nullptr) {
      return Next::kStop;
    }
    deserializeEvent(*eview);
    return Next::kEvent;
  }

}  // namespace edm

#endif
