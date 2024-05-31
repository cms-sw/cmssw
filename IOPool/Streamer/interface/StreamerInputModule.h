#ifndef IOPool_Streamer_StreamerInputModule_h
#define IOPool_Streamer_StreamerInputModule_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "IOPool/Streamer/interface/EventMessage.h"

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

namespace edm::streamer {
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
      if (didArtificialFile_) {
        didArtificialFile_ = false;

        return;
      }
      if (pr_.get() != nullptr)
        pr_->closeFile();
    }

    void setupMetaData() {
      InitMsgView const* header = pr_->getHeader();
      assert(header);
      deserializeAndMergeWithRegistry(*header);

      //NOTE: should read first Event to get the meta data
      auto eview = pr_->getNextEvent();
      assert(eview);
      assert(eview->isEventMetaData());
      deserializeEventMetaData(*eview);
      updateEventMetaData();
    }

    void genuineReadFile() override {
      if (isFirstFile_) {
        isFirstFile_ = false;
        return;
      }

      if (didArtificialFile_) {
        //update the event meta data
        didArtificialFile_ = false;
        updateEventMetaData();

        return;
      }
      setupMetaData();
    }

    Next checkNext() override;

    edm::propagate_const<std::unique_ptr<Producer>> pr_;
    bool isFirstFile_ = true;
    bool didArtificialFile_ = false;
  };  //end-of-class-def

  template <typename Producer>
  StreamerInputModule<Producer>::~StreamerInputModule() {}

  template <typename Producer>
  StreamerInputModule<Producer>::StreamerInputModule(ParameterSet const& pset, InputSourceDescription const& desc)
      : StreamerInputSource(pset, desc),
        //prod_reg_(&productRegistry()),
        pr_(new Producer(pset)) {
    //Get header/init from Producer
    setupMetaData();
  }

  template <typename Producer>
  StreamerInputSource::Next StreamerInputModule<Producer>::checkNext() {
    EventMsgView const* eview = pr_->getNextEvent();

    if (eview == nullptr) {
      if (pr_->newHeader()) {
        FDEBUG(6) << "A new file has been opened and we must compare Headers here !!" << std::endl;
        return Next::kFile;
      }
      return Next::kStop;
    }
    if (eview->isEventMetaData()) {
      //we lie and say there is a new file since we need to synchronize to update the meta data
      deserializeEventMetaData(*eview);
      didArtificialFile_ = true;
      return Next::kFile;
    }
    deserializeEvent(*eview);
    return Next::kEvent;
  }

}  // namespace edm::streamer

#endif
