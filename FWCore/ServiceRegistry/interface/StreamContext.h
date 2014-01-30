#ifndef FWCore_ServiceRegistry_StreamContext_h
#define FWCore_ServiceRegistry_StreamContext_h

/**\class edm::StreamContext

 Description: Holds pointer to ProcessContext, StreamID,
 transition, EventID and timestamp.
 This is intended primarily to be passed to Services
 as an argument to their callback functions.

 Usage:


*/
//
// Original Author: W. David Dagenhart
//         Created: 7/8/2013

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <iosfwd>

namespace edm {

  class ProcessContext;

  class StreamContext {

  public:

    enum class Transition {
      kBeginStream,
      kBeginRun,
      kBeginLuminosityBlock,
      kEvent,
      kEndLuminosityBlock,
      kEndRun,
      kEndStream,
      kInvalid
    };

    StreamContext(StreamID const& streamID,
                  ProcessContext const* processContext);
    
    StreamContext(StreamID const& streamID,
                  Transition transition,
                  EventID const& eventID,
                  RunIndex const& runIndex,
                  LuminosityBlockIndex const& luminosityBlockIndex, 
                  Timestamp const & timestamp,
                  ProcessContext const* processContext);

    StreamID const& streamID() const { return streamID_; }
    Transition transition() const { return transition_; }
    EventID const& eventID() const { return eventID_; } // event#==0 is a lumi, event#==0&lumi#==0 is a run
    RunIndex const& runIndex() const { return runIndex_; }
    LuminosityBlockIndex const& luminosityBlockIndex() const { return luminosityBlockIndex_; }
    Timestamp const& timestamp() const { return timestamp_; }
    ProcessContext const* processContext() const { return processContext_; }

    void setTransition(Transition v) { transition_ = v; }
    void setEventID(EventID const& v) { eventID_ = v; }
    void setRunIndex(RunIndex const& v) { runIndex_ = v; }
    void setLuminosityBlockIndex(LuminosityBlockIndex const& v) { luminosityBlockIndex_ = v; }
    void setTimestamp(Timestamp const& v) { timestamp_ = v; }

  private:
    StreamID streamID_;
    Transition transition_;
    EventID eventID_; // event#==0 is a lumi, event#==0&lumi#==0 is a run
    RunIndex runIndex_;
    LuminosityBlockIndex luminosityBlockIndex_; 
    Timestamp timestamp_;
    ProcessContext const* processContext_;
  };

  std::ostream& operator<<(std::ostream&, StreamContext const&);
}
#endif
