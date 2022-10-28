#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"

#include <ostream>

namespace edm {

  GlobalContext::GlobalContext(Transition transition,
                               LuminosityBlockID const& luminosityBlockID,
                               RunIndex const& runIndex,
                               LuminosityBlockIndex const& luminosityBlockIndex,
                               Timestamp const& timestamp,
                               ProcessContext const* processContext)
      : transition_(transition),
        luminosityBlockID_(luminosityBlockID),
        runIndex_(runIndex),
        luminosityBlockIndex_(luminosityBlockIndex),
        timestamp_(timestamp),
        processContext_(processContext) {}

  GlobalContext::GlobalContext(Transition transition, ProcessContext const* processContext)
      : transition_(transition),
        luminosityBlockID_(),
        runIndex_(RunIndex::invalidRunIndex()),
        luminosityBlockIndex_(LuminosityBlockIndex::invalidLuminosityBlockIndex()),
        timestamp_(),
        processContext_(processContext) {}

  std::ostream& operator<<(std::ostream& os, GlobalContext const& gc) {
    os << "GlobalContext: transition = ";
    switch (gc.transition()) {
      case GlobalContext::Transition::kBeginJob:
        os << "BeginJob";
        break;
      case GlobalContext::Transition::kBeginProcessBlock:
        os << "BeginProcessBlock";
        break;
      case GlobalContext::Transition::kAccessInputProcessBlock:
        os << "AccessInputProcessBlock";
        break;
      case GlobalContext::Transition::kBeginRun:
        os << "BeginRun";
        break;
      case GlobalContext::Transition::kBeginLuminosityBlock:
        os << "BeginLuminosityBlock";
        break;
      case GlobalContext::Transition::kEndLuminosityBlock:
        os << "EndLuminosityBlock";
        break;
      case GlobalContext::Transition::kEndRun:
        os << "EndRun";
        break;
      case GlobalContext::Transition::kEndProcessBlock:
        os << "EndProcessBlock";
        break;
      case GlobalContext::Transition::kEndJob:
        os << "EndJob";
        break;
      case GlobalContext::Transition::kWriteProcessBlock:
        os << "WriteProcessBlock";
        break;
      case GlobalContext::Transition::kWriteRun:
        os << "WriteRun";
        break;
      case GlobalContext::Transition::kWriteLuminosityBlock:
        os << "WriteLuminosityBlock";
        break;
    }
    os << "\n    " << gc.luminosityBlockID() << "\n    runIndex = " << gc.runIndex().value()
       << "  luminosityBlockIndex = " << gc.luminosityBlockIndex().value()
       << "  unixTime = " << gc.timestamp().unixTime() << " microsecondOffset = " << gc.timestamp().microsecondOffset()
       << "\n";
    if (gc.processContext()) {
      os << "    " << *gc.processContext();
    }
    return os;
  }

  void exceptionContext(std::ostream& os, GlobalContext const& gc) {
    os << "Processing ";
    switch (gc.transition()) {
      case GlobalContext::Transition::kBeginJob:
        os << "begin Job";
        break;
      case GlobalContext::Transition::kBeginProcessBlock:
        os << "begin ProcessBlock";
        break;
      case GlobalContext::Transition::kAccessInputProcessBlock:
        os << "access input ProcessBlock";
        break;
      case GlobalContext::Transition::kBeginRun:
        os << "global begin Run " << RunID(gc.luminosityBlockID().run());
        break;
      case GlobalContext::Transition::kBeginLuminosityBlock:
        os << "global begin LuminosityBlock " << gc.luminosityBlockID();
        break;
      case GlobalContext::Transition::kEndLuminosityBlock:
        os << "global end LuminosityBlock " << gc.luminosityBlockID();
        break;
      case GlobalContext::Transition::kEndRun:
        os << "global end Run " << RunID(gc.luminosityBlockID().run());
        break;
      case GlobalContext::Transition::kEndProcessBlock:
        os << "end ProcessBlock";
        break;
      case GlobalContext::Transition::kEndJob:
        os << "endJob";
        break;
      case GlobalContext::Transition::kWriteProcessBlock:
        os << "write ProcessBlock";
        break;
      case GlobalContext::Transition::kWriteRun:
        os << "write Run " << RunID(gc.luminosityBlockID().run());
        break;
      case GlobalContext::Transition::kWriteLuminosityBlock:
        os << "write LuminosityBlock " << gc.luminosityBlockID();
        break;
    }
  }

  std::string_view transitionName(GlobalContext::Transition iTrans) {
    switch (iTrans) {
      case GlobalContext::Transition::kBeginJob:
        return "begin Job";
      case GlobalContext::Transition::kBeginProcessBlock:
        return "begin ProcessBlock";
      case GlobalContext::Transition::kAccessInputProcessBlock:
        return "access input ProcessBlock";
      case GlobalContext::Transition::kBeginRun:
        return "global begin Run";
      case GlobalContext::Transition::kBeginLuminosityBlock:
        return "global begin LuminosityBlock";
      case GlobalContext::Transition::kEndLuminosityBlock:
        return "global end LuminosityBlock";
      case GlobalContext::Transition::kEndRun:
        return "global end Run";
      case GlobalContext::Transition::kEndProcessBlock:
        return "end ProcessBlock";
      case GlobalContext::Transition::kEndJob:
        return "endJob";
      case GlobalContext::Transition::kWriteProcessBlock:
        return "write ProcessBlock";
      case GlobalContext::Transition::kWriteRun:
        return "write Run";
      case GlobalContext::Transition::kWriteLuminosityBlock:
        return "write LuminosityBlock";
    }
    return "Unknown";
  }
}  // namespace edm
