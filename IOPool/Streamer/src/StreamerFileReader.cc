#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/src/StreamerFileReader.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"

#include <cassert>
#include <iterator>
#include <utility>
namespace edm::streamer {

  namespace {
    // numDecodeTasks: 0 = as many as the framework streams, 1 = serial decoding, N > 1 = N.
    // Returns the size of the decoding window, 0 for serial decoding.
    unsigned int decodeWindowSize(unsigned int numDecodeTasks, InputSourceDescription const& desc) {
      if (numDecodeTasks == 0) {
        numDecodeTasks = desc.allocations_ ? desc.allocations_->numberOfStreams() : 1;
      }
      // with a single event read ahead, the source would almost always decode it itself
      return numDecodeTasks > 1 ? numDecodeTasks : 0;
    }
  }  // namespace

  StreamerFileReader::StreamerFileReader(ParameterSet const& pset, InputSourceDescription const& desc)
      : StreamerInputSource(pset, desc),
        inputFileCatalog_(pset),
        streamReader_(),
        eventSkipperByID_(EventSkipperByID::create(pset).release()),
        initialNumberOfEventsToSkip_(pset.getUntrackedParameter<unsigned int>("skipEvents")),
        prefetchMBytes_(pset.getUntrackedParameter<unsigned int>("prefetchMBytes")),
        decodeWindowSize_(decodeWindowSize(pset.getUntrackedParameter<unsigned int>("numDecodeTasks"), desc)) {
    reset_();
  }

  StreamerFileReader::~StreamerFileReader() { drainDecodeWindow(); }

  void StreamerFileReader::reset_() {
    drainDecodeWindow();
    auto const& configuredFileNames = inputFileCatalog_.configuredFileNames();
    if (configuredFileNames.size() > 1u) {
      streamReader_ = std::make_unique<StreamerInputFile>(inputFileCatalog_, eventSkipperByID(), prefetchMBytes_);
    } else if (configuredFileNames.size() == 1u) {
      streamReader_ = std::make_unique<StreamerInputFile>(inputFileCatalog_.firstPFNFromFirstCatalog(),
                                                          inputFileCatalog_.logicalFileName(0),
                                                          eventSkipperByID(),
                                                          prefetchMBytes_);
    } else {
      throw Exception(errors::FileReadError, "StreamerFileReader::StreamerFileReader")
          << "No fileNames were specified\n";
    }
    isFirstFile_ = true;
    updateMetaData(false);
    if (initialNumberOfEventsToSkip_) {
      skip(initialNumberOfEventsToSkip_);
    }
  }

  void StreamerFileReader::updateMetaData(bool subsequent) {
    InitMsgView const* header = getHeader();
    deserializeAndMergeWithRegistry(*header, subsequent);
    //NOTE: should read first Event to get the meta data and then set 'artificial file'
    auto eview = getNextEvent();

    //if no events then file must be empty
    if (eview == nullptr)
      return;

    assert(eview->isEventMetaData());
    deserializeEventMetaData(*eview);
    updateEventMetaData();
  }

  StreamerFileReader::Next StreamerFileReader::checkNext() {
    return decodeWindowSize_ == 0 ? checkNextSerial() : checkNextPipelined();
  }

  StreamerFileReader::Next StreamerFileReader::checkNextPipelined() {
    decodeAhead();
    if (decodeWindow_.empty()) {
      // all the events read before the barrier have been handed out
      Barrier barrier = std::exchange(barrier_, Barrier::kNone);
      if (barrier == Barrier::kError) {
        std::rethrow_exception(std::exchange(readError_, nullptr));
      }
      if (barrier == Barrier::kNewMetaData) {
        //we lie and say there is a new file since we need to synchronize to update the meta data
        didArtificialFile_ = true;
        deserializeEventMetaData(*streamReader_->currentRecord());
        return Next::kFile;
      }
      if (newHeader()) {
        return Next::kFile;
      }
      return Next::kStop;
    }

    waitForFront();
    std::unique_ptr<DecodeSlot> slot = std::move(decodeWindow_.front());
    decodeWindow_.pop_front();
    std::exception_ptr exception = std::exchange(slot->exception, nullptr);
    DecodedEvent decoded = std::move(slot->decoded);
    releaseSlot(std::move(slot));
    if (exception) {
      std::rethrow_exception(exception);
    }
    finishEvent(std::move(decoded));

    // keep the window full while the framework processes this event
    decodeAhead();
    return Next::kEvent;
  }

  void StreamerFileReader::decodeSlot(DecodeSlot& slot) const {
    try {
      EventMsgView view(slot.message.data());
      decodeEvent(view, slot.dest, slot.decoded);
    } catch (...) {
      // rethrown when the event is reached, to keep the error in input order
      slot.exception = std::current_exception();
    }
  }

  bool StreamerFileReader::claimAndDecode(DecodeSlot& slot, std::atomic<int>& state) const {
    int expected = DecodeSlot::kQueued;
    if (not state.compare_exchange_strong(expected, DecodeSlot::kClaimed)) {
      // claimed by someone else: do not touch the slot, it may already be in use for another event
      return false;
    }
    decodeSlot(slot);
    state.store(DecodeSlot::kDone, std::memory_order_release);
    // only the source waits for an event; the state outlives the slot's use
    state.notify_one();
    return true;
  }

  // an event that will not be used: if its decoding has not started, mark it as done so that its task
  // returns without decoding it; if its task is decoding it, wait until it is done with the slot
  void StreamerFileReader::discard(DecodeSlot& slot) {
    std::atomic<int>& state = *slot.state;
    int current = DecodeSlot::kQueued;
    if (state.compare_exchange_strong(current, DecodeSlot::kDone)) {
      return;
    }
    while (current != DecodeSlot::kDone) {
      state.wait(current, std::memory_order_acquire);
      current = state.load(std::memory_order_acquire);
    }
  }

  void StreamerFileReader::waitForFront() {
    DecodeSlot& front = *decodeWindow_.front();
    std::atomic<int>& state = *front.state;
    if (claimAndDecode(front, state)) {
      return;
    }
    // the first event is being decoded by its task: meanwhile, decode the next ones, in order,
    // then wait for it without using a core (std::atomic::wait spins briefly, then sleeps)
    auto next = std::next(decodeWindow_.begin());
    int current = state.load(std::memory_order_acquire);
    while (current != DecodeSlot::kDone) {
      if (next != decodeWindow_.end()) {
        claimAndDecode(**next, *(*next)->state);
        ++next;
      } else {
        state.wait(current, std::memory_order_acquire);
      }
      current = state.load(std::memory_order_acquire);
    }
  }

  void StreamerFileReader::decodeAhead() {
    if (not arena_) {
      // checkNext() runs in the framework's arena
      arena_ = std::make_unique<oneapi::tbb::task_arena>(oneapi::tbb::task_arena::attach());
    }
    while (barrier_ == Barrier::kNone and decodeWindow_.size() < decodeWindowSize_) {
      EventMsgView const* eview = nullptr;
      try {
        eview = getNextEvent();
      } catch (...) {
        // rethrown once the events read before the error have been handed out, as in serial mode
        readError_ = std::current_exception();
        barrier_ = Barrier::kError;
        return;
      }
      if (eview == nullptr) {
        barrier_ = Barrier::kEndOfInput;
        return;
      }
      if (eview->isEventMetaData()) {
        if (presentEventMetaDataChecksum() != eventMetaDataChecksum(*eview)) {
          // stop here, the message is still available through streamReader_->currentRecord()
          barrier_ = Barrier::kNewMetaData;
          return;
        }
        //skip this meta data
        continue;
      }

      std::unique_ptr<DecodeSlot> slot = takeSlot();
      slot->message.assign(eview->startAddress(), eview->startAddress() + eview->size());
      // if the tasks spawned earlier are still waiting for a thread, the source decodes this event itself
      if (pendingTasks_.load(std::memory_order_relaxed) < 4 * decodeWindowSize_) {
        pendingTasks_.fetch_add(1, std::memory_order_relaxed);
        group_.run([this, s = slot.get(), state = slot->state]() {
          pendingTasks_.fetch_sub(1, std::memory_order_relaxed);
          claimAndDecode(*s, *state);
        });
      }
      decodeWindow_.push_back(std::move(slot));
    }
  }

  std::unique_ptr<StreamerFileReader::DecodeSlot> StreamerFileReader::takeSlot() {
    std::unique_ptr<DecodeSlot> slot;
    if (freeSlots_.empty()) {
      slot = std::make_unique<DecodeSlot>();
    } else {
      slot = std::move(freeSlots_.back());
      freeSlots_.pop_back();
    }
    slot->decoded = DecodedEvent();
    slot->exception = nullptr;
    // a new state for each event: a task still holding the previous one will not touch the slot
    slot->state = std::make_shared<std::atomic<int>>(DecodeSlot::kQueued);
    return slot;
  }

  // the slot's event must be done (decoded and taken, or discarded): no task uses the slot any more
  void StreamerFileReader::releaseSlot(std::unique_ptr<DecodeSlot> slot) {
    if (freeSlots_.size() < decodeWindowSize_) {
      // release the products of a skipped event, keep the buffers for reuse
      slot->decoded = DecodedEvent();
      slot->state.reset();
      freeSlots_.push_back(std::move(slot));
    }
    // otherwise the slot is destroyed: do not keep more buffers than the window needs
  }

  void StreamerFileReader::drainDecodeWindow() {
    if (arena_) {
      // the events read ahead will not be used: their tasks that have not started will return without
      // decoding them; wait for all the tasks from within the arena, so that this thread can run the
      // pending ones itself
      for (auto& slot : decodeWindow_) {
        discard(*slot);
      }
      decodeWindow_.clear();
      arena_->execute([this]() { group_.wait(); });
    }
    barrier_ = Barrier::kNone;
    readError_ = nullptr;
  }

  StreamerFileReader::Next StreamerFileReader::checkNextSerial() {
    EventMsgView const* eview = getNextEvent();

    if (eview == nullptr) {
      if (newHeader()) {
        return Next::kFile;
      }
      return Next::kStop;
    }
    if (eview->isEventMetaData()) {
      if (presentEventMetaDataChecksum() != eventMetaDataChecksum(*eview)) {
        //we lie and say there is a new file since we need to synchronize to update the meta data
        didArtificialFile_ = true;
        deserializeEventMetaData(*eview);
        return Next::kFile;
      } else {
        //skip this meta data
        eview = getNextEvent();
        if (eview == nullptr) {
          if (newHeader()) {
            return Next::kFile;
          }
          return Next::kStop;
        }
      }
    }
    deserializeEvent(*eview);
    return Next::kEvent;
  }

  void StreamerFileReader::skip(int toSkip) {
    // the events already read ahead come first: they are not decoded if their task has not started
    // yet, otherwise the source waits for their task to finish, and the result is dropped
    while (toSkip > 0 and not decodeWindow_.empty()) {
      std::unique_ptr<DecodeSlot> slot = std::move(decodeWindow_.front());
      decodeWindow_.pop_front();
      discard(*slot);
      releaseSlot(std::move(slot));
      --toSkip;
    }
    if (toSkip == 0 or barrier_ != Barrier::kNone) {
      // keep a pending barrier: new event meta data and read errors must be handled by checkNext().
      // Skipping stops at the end of a file, as in serial mode, and also at new event meta data,
      // where serial mode would read past the meta data message without handling it.
      return;
    }
    for (int i = 0; i != toSkip; ++i) {
      EventMsgView const* evMsg = getNextEvent();
      if (evMsg == nullptr) {
        return;
      }
      // If the event would have been skipped anyway, don't count it as a skipped event.
      if (eventSkipperByID_ && eventSkipperByID_->skipIt(evMsg->run(), evMsg->lumi(), evMsg->event())) {
        --i;
      }
    }
  }

  void StreamerFileReader::genuineCloseFile() {
    if (didArtificialFile_) {
      return;
    }
    if (streamReader_.get() != nullptr)
      streamReader_->closeStreamerFile();
  }

  void StreamerFileReader::genuineReadFile() {
    if (isFirstFile_) {
      //The file was already opened in the constructor
      isFirstFile_ = false;
      return;
    }
    if (didArtificialFile_) {
      //update the event meta data
      didArtificialFile_ = false;
      updateEventMetaData();
      return;
    }
    streamReader_->openNextFile();
    // A new file has been opened and we must compare Heraders here !!
    //Get header/init from reader
    updateMetaData(true);
  }

  bool StreamerFileReader::newHeader() { return streamReader_->newHeader(); }

  InitMsgView const* StreamerFileReader::getHeader() {
    InitMsgView const* header = streamReader_->startMessage();

    if (header->code() != Header::INIT) {  //INIT Msg
      throw Exception(errors::FileReadError, "StreamerFileReader::readHeader")
          << "received wrong message type: expected INIT, got " << header->code() << "\n";
    }
    return header;
  }

  EventMsgView const* StreamerFileReader::getNextEvent() {
    if (StreamerInputFile::Next::kEvent != streamReader_->next()) {
      return nullptr;
    }
    return streamReader_->currentRecord();
  }

  void StreamerFileReader::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Reads events from streamer files.");
    InputFileCatalog::fillDescription(desc);
    desc.addUntracked<unsigned int>("skipEvents", 0U)
        ->setComment("Skip the first 'skipEvents' events that otherwise would have been processed.");
    //This next parameter is read in the base class, but its default value depends on the derived class, so it is set here.
    desc.addUntracked<bool>("inputFileTransitionsEachEvent", false);
    desc.addUntracked<unsigned int>("prefetchMBytes", 0);
    desc.addUntracked<unsigned int>("numDecodeTasks", 1)
        ->setComment(
            "Number of events read ahead and decoded (checksum, uncompression, deserialization) concurrently by "
            "TBB tasks, and handed to the framework in input order. 1 (the default) decodes each event serially "
            "when the framework asks for it; N > 1 reads up to N events ahead; 0 reads as many events ahead as the "
            "number of framework streams. Each event read ahead keeps a copy of its message, its uncompression "
            "buffer and its deserialized products in memory until the framework takes it, so the memory needed "
            "grows with N (with 0, roughly by one more set of input products per stream).");
    StreamerInputSource::fillDescription(desc);
    EventSkipperByID::fillDescription(desc);
    descriptions.add("source", desc);
  }
}  // namespace edm::streamer
