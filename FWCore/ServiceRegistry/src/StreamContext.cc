#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

#include <ostream>

namespace edm {

  StreamContext::StreamContext(StreamID const& streamID, ProcessContext const* processContext)
      : streamID_(streamID),
        transition_(Transition::kInvalid),
        eventID_(EventID(0, 0, 0)),
        runIndex_(RunIndex::invalidRunIndex()),
        luminosityBlockIndex_(LuminosityBlockIndex::invalidLuminosityBlockIndex()),
        timestamp_(),
        processContext_(processContext) {}

  StreamContext::StreamContext(StreamID const& streamID,
                               Transition transition,
                               EventID const& eventID,
                               RunIndex const& runIndex,
                               LuminosityBlockIndex const& luminosityBlockIndex,
                               Timestamp const& timestamp,
                               ProcessContext const* processContext)
      : streamID_(streamID),
        transition_(transition),
        eventID_(eventID),
        runIndex_(runIndex),
        luminosityBlockIndex_(luminosityBlockIndex),
        timestamp_(timestamp),
        processContext_(processContext) {}

  std::ostream& operator<<(std::ostream& os, StreamContext const& sc) {
    os << "StreamContext: StreamID = " << sc.streamID() << " transition = ";
    switch (sc.transition()) {
      case StreamContext::Transition::kBeginStream:
        os << "BeginStream";
        break;
      case StreamContext::Transition::kBeginRun:
        os << "BeginRun";
        break;
      case StreamContext::Transition::kBeginLuminosityBlock:
        os << "BeginLuminosityBlock";
        break;
      case StreamContext::Transition::kEvent:
        os << "Event";
        break;
      case StreamContext::Transition::kEndLuminosityBlock:
        os << "EndLuminosityBlock";
        break;
      case StreamContext::Transition::kEndRun:
        os << "EndRun";
        break;
      case StreamContext::Transition::kEndStream:
        os << "EndStream";
        break;
      case StreamContext::Transition::kInvalid:
        os << "Invalid";
        break;
    }
    os << "\n    " << sc.eventID() << "\n    runIndex = " << sc.runIndex().value()
       << "  luminosityBlockIndex = " << sc.luminosityBlockIndex().value()
       << "  unixTime = " << sc.timestamp().unixTime() << " microsecondOffset = " << sc.timestamp().microsecondOffset()
       << "\n";
    if (sc.processContext()) {
      os << "    " << *sc.processContext();
    }
    return os;
  }

  void exceptionContext(std::ostream& os, StreamContext const& sc) {
    os << "Processing  ";
    auto id = sc.eventID();
    switch (sc.transition()) {
      case StreamContext::Transition::kBeginStream:
        os << "begin Stream";
        break;
      case StreamContext::Transition::kBeginRun:
        os << "stream begin Run " << RunID(id.run());
        break;
      case StreamContext::Transition::kBeginLuminosityBlock:
        os << "stream begin LuminosityBlock " << LuminosityBlockID(id.run(), id.luminosityBlock());
        break;
      case StreamContext::Transition::kEvent:
        os << "Event " << sc.eventID();
        break;
      case StreamContext::Transition::kEndLuminosityBlock:
        os << "stream end LuminosityBlock " << LuminosityBlockID(id.run(), id.luminosityBlock());
        break;
      case StreamContext::Transition::kEndRun:
        os << "stream end Run " << RunID(id.run());
        break;
      case StreamContext::Transition::kEndStream:
        os << "end Stream";
        break;
      case StreamContext::Transition::kInvalid:
        os << "Invalid";
        break;
    }
    os << " stream: " << sc.streamID();
  }

  std::string_view transitionName(StreamContext::Transition iTrans) {
    switch (iTrans) {
      case StreamContext::Transition::kBeginStream:
        return "begin Stream";
      case StreamContext::Transition::kBeginRun:
        return "stream begin Run";
      case StreamContext::Transition::kBeginLuminosityBlock:
        return "stream begin LuminosityBlock";
      case StreamContext::Transition::kEvent:
        return "Event";
      case StreamContext::Transition::kEndLuminosityBlock:
        return "stream end LuminosityBlock";
      case StreamContext::Transition::kEndRun:
        return "stream end Run";
      case StreamContext::Transition::kEndStream:
        return "end Stream";
      case StreamContext::Transition::kInvalid:
        return "Invalid";
    }
    return "Unknown";
  }

}  // namespace edm
