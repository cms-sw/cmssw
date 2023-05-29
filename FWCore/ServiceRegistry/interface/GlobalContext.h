#ifndef FWCore_ServiceRegistry_GlobalContext_h
#define FWCore_ServiceRegistry_GlobalContext_h

/**\class edm::GlobalContext

 Description: This is intended primarily to be passed to
Services as an argument to their callback functions. It contains
information about the current state of global processing.

 Usage:


*/
//
// Original Author: W. David Dagenhart
//         Created: 7/10/2013

#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/RunIndex.h"

#include <iosfwd>
#include <string_view>

namespace edm {

  class ProcessContext;

  class GlobalContext {
  public:
    enum class Transition {
      kBeginJob,
      kBeginProcessBlock,
      kAccessInputProcessBlock,
      kBeginRun,
      kBeginLuminosityBlock,
      kEndLuminosityBlock,
      kEndRun,
      kEndProcessBlock,
      kEndJob,
      kWriteProcessBlock,
      kWriteRun,
      kWriteLuminosityBlock
    };

    GlobalContext(Transition transition,
                  LuminosityBlockID const& luminosityBlockID,
                  RunIndex const& runIndex,
                  LuminosityBlockIndex const& luminosityBlockIndex,
                  Timestamp const& timestamp,
                  ProcessContext const* processContext);
    GlobalContext(Transition transition, ProcessContext const* processContext);

    Transition transition() const { return transition_; }
    bool isAtEndTransition() const {
      return transition() == Transition::kEndLuminosityBlock or transition() == Transition::kEndRun or
             transition() == Transition::kEndProcessBlock or transition() == Transition::kWriteRun or
             transition() == Transition::kWriteLuminosityBlock or transition() == Transition::kWriteProcessBlock;
    }

    LuminosityBlockID const& luminosityBlockID() const { return luminosityBlockID_; }
    RunIndex const& runIndex() const { return runIndex_; }
    LuminosityBlockIndex const& luminosityBlockIndex() const { return luminosityBlockIndex_; }
    Timestamp const& timestamp() const { return timestamp_; }
    ProcessContext const* processContext() const { return processContext_; }

  private:
    Transition transition_;
    LuminosityBlockID luminosityBlockID_;
    RunIndex runIndex_;
    LuminosityBlockIndex luminosityBlockIndex_;
    Timestamp timestamp_;
    ProcessContext const* processContext_;
  };

  void exceptionContext(std::ostream&, GlobalContext const&);
  std::ostream& operator<<(std::ostream&, GlobalContext const&);

  std::string_view transitionName(GlobalContext::Transition);
}  // namespace edm
#endif
