#ifndef IOPool_Streamer_StreamerFileReader_h
#define IOPool_Streamer_StreamerFileReader_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWStorage/Catalog/interface/InputFileCatalog.h"

#include <atomic>
#include <deque>
#include <exception>
#include <memory>
#include <vector>

#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

class EventMsgView;
class InitMsgView;

namespace edm {
  class ConfigurationDescriptions;
  class EventPrincipal;
  class EventSkipperByID;
  struct InputSourceDescription;
  class ParameterSet;
  namespace streamer {
    class StreamerInputFile;
    class StreamerFileReader : public StreamerInputSource {
    public:
      StreamerFileReader(ParameterSet const& pset, InputSourceDescription const& desc);
      ~StreamerFileReader() override;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

    private:
      InitMsgView const* getHeader();
      EventMsgView const* getNextEvent();
      bool newHeader();
      void updateMetaData(bool subsequent);

      Next checkNext() override;
      Next checkNextSerial();
      Next checkNextPipelined();
      void skip(int toSkip) override;
      void genuineReadFile() override;
      void genuineCloseFile() override;
      void reset_() override;

      std::shared_ptr<EventSkipperByID const> eventSkipperByID() const {
        return get_underlying_safe(eventSkipperByID_);
      }
      std::shared_ptr<EventSkipperByID>& eventSkipperByID() { return get_underlying_safe(eventSkipperByID_); }

      edm::InputFileCatalog inputFileCatalog_;
      edm::propagate_const<std::unique_ptr<StreamerInputFile>> streamReader_;
      edm::propagate_const<std::shared_ptr<EventSkipperByID>> eventSkipperByID_;
      int initialNumberOfEventsToSkip_;
      int prefetchMBytes_;
      bool isFirstFile_ = true;
      bool didArtificialFile_ = false;

      // Pipelined decoding: up to decodeWindowSize_ events are read ahead, in input order, and
      // decoded concurrently by TBB tasks; checkNext() hands them to the framework in order.
      // An event is decoded either by its TBB task or, if that has not started yet when the event is
      // needed, by the source itself: whoever claims it first. The source never waits on the task
      // group, which could make it run unrelated (and long) tasks while the streams need events.
      // The task and the source compete through a state shared with the task, created anew for each
      // event: the one that loses never touches the slot, so a slot can be reused as soon as its event
      // is done, even if its task has not run yet (with one thread it may never run before the end).
      struct DecodeSlot {
        enum State { kQueued, kClaimed, kDone };
        std::vector<unsigned char> message;  // copy of the event message
        std::vector<unsigned char> dest;     // uncompression buffer
        DecodedEvent decoded;
        std::exception_ptr exception;
        std::shared_ptr<std::atomic<int>> state;
      };
      // Where reading ahead stopped: the end of the current file or of the input, an event meta
      // data message with a new checksum, or an error reading the input; they must be processed
      // serially, once all the events read before them have been handed out.
      enum class Barrier { kNone, kEndOfInput, kNewMetaData, kError };

      void decodeAhead();
      void decodeSlot(DecodeSlot& slot) const;
      bool claimAndDecode(DecodeSlot& slot, std::atomic<int>& state) const;
      static void discard(DecodeSlot& slot);
      void waitForFront();
      void drainDecodeWindow();
      std::unique_ptr<DecodeSlot> takeSlot();
      void releaseSlot(std::unique_ptr<DecodeSlot> slot);

      unsigned int decodeWindowSize_;  // 0 for serial decoding
      // the arena where the decoding tasks run (the framework's one), so that they can be
      // waited for also from outside of it, e.g. when the source is destroyed
      std::unique_ptr<oneapi::tbb::task_arena> arena_;
      // all the decoding tasks; waited for only when the window is drained
      oneapi::tbb::task_group group_;
      // decoding tasks spawned that have not started yet; above a few times the window size no more
      // are spawned, and the source decodes the events itself
      std::atomic<unsigned int> pendingTasks_ = 0;
      std::deque<std::unique_ptr<DecodeSlot>> decodeWindow_;
      std::vector<std::unique_ptr<DecodeSlot>> freeSlots_;
      Barrier barrier_ = Barrier::kNone;
      std::exception_ptr readError_;  // for Barrier::kError
    };
  }  // namespace streamer
}  // namespace edm

#endif
