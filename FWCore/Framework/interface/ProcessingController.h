#ifndef FWCore_Framework_ProcessingController_h
#define FWCore_Framework_ProcessingController_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProcessingController
//
/**\class ProcessingController ProcessingController.h FWCore/Framework/interface/ProcessingController.h

 Description: Interface used by an EDLooper to specify what event we should process next

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Aug  6 16:06:01 CDT 2010
//

// system include files

// user include files
#include "DataFormats/Provenance/interface/EventID.h"

// forward declarations
namespace edm {

  class ProcessingController {
  public:
    enum ForwardState {
      kEventsAheadInFile,
      kNextFileExists,  // but no events ahead in this file
      kAtLastEvent,
      kUnknownForward  // returned by a source which is not random accessible
    };

    enum ReverseState {
      kEventsBackwardsInFile,
      kPreviousFileExists,  // but no events backwards in this file
      kAtFirstEvent,
      kUnknownReverse  // returned by a source which is not random accessible
    };

    ProcessingController(ForwardState forwardState, ReverseState reverseState, bool iCanRandomAccess);
    ProcessingController(const ProcessingController&) = delete;                   // stop default
    const ProcessingController& operator=(const ProcessingController&) = delete;  // stop default

    // ---------- const member functions ---------------------

    ///Returns the present state of processing
    ForwardState forwardState() const;
    ReverseState reverseState() const;

    ///Returns 'true' if the job's source can randomly access
    bool canRandomAccess() const;

    enum Transition { kToNextEvent, kToPreviousEvent, kToSpecifiedEvent };

    Transition requestedTransition() const;

    ///If 'setTransitionToEvent was called this returns the value passed,
    /// else it returns an invalid EventID
    edm::EventID specifiedEventTransition() const;

    bool lastOperationSucceeded() const;

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

    /** Tells the framework that we should go onto the next event in the sequence.
       If there is no next event the job will drop out of the event loop.
       */
    void setTransitionToNextEvent();

    /** Tells the framework we should backup and run the previous event seen in the sequence.
       If you are already at the first event the job will drop out of the event loop
       */
    void setTransitionToPreviousEvent();

    /** Tells the framework that the next event to processes should be iID.
       If event iID can not be found in the source, the job will drop out of the event loop.
       */
    void setTransitionToEvent(edm::EventID const& iID);

    void setLastOperationSucceeded(bool value);

  private:
    // ---------- member data --------------------------------
    ForwardState forwardState_;
    ReverseState reverseState_;
    Transition transition_;
    EventID specifiedEvent_;
    bool canRandomAccess_;
    bool lastOperationSucceeded_;
  };
}  // namespace edm
#endif
