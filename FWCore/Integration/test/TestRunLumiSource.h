#ifndef Framework_TestRunLumiSource_h
#define Framework_TestRunLumiSource_h

/*----------------------------------------------------------------------
$Id: TestRunLumiSource.h,v 1.6 2008/07/31 23:10:46 wmtan Exp $

This source is intended only for test purposes.  With it one can
create data files with arbitrary sequences of run number, lumi
number, and event number in the auxiliary objects in the run tree,
lumi tree, and event tree.  It is quite possible to create an illegal
format that cannot be read with any input module using this source.

The output files of jobs using this source will be used in tests of
input modules to verify they are behaving properly.

The configuration looks as follows

  source = TestRunLumiSource {
    untracked vint32 runLumiEvent = { 1, 1, 1,    # run
                                      1, 1, 1,    # lumi
                                      1, 1, 1,    # event
                                      1, 1, 2,    # event
                                      0, 0, 0,    # causes end lumi
                                      0, 0, 0     # causes end run
                                    }
  }

Each line contains 3 values: run, lumi, and event.  These lines
are used in order, one line per each call to readRun_,
readLuminosityBlock_, and readEvent, in the order called by the
event processor.  Note that when readRun_ is called only the run
number is used and the other two values are extraneous.  When
readLuminosityBlock is called only the first two values are used.
(0, 0, 0) will trigger the end of the current luminosity block,
run, or job as appropriate for when it appears. Running off the
bottom list is also equivalent to (0,0,0). What is shown above
is the typical sequence one would expect for two events, but this
source is capable of handling arbitrary sequences of run numbers,
lumi block number, and events.  For test purposes one can even
include sequences that make no sense and the entries will get
written to the output file anyway.

----------------------------------------------------------------------*/


#include "FWCore/Framework/interface/InputSource.h"

#include "boost/shared_ptr.hpp"
#include <memory>
#include <vector>

namespace edm {

  class ParameterSet;
  class InputSourceDescription;
  class EventPrincipal;
  class LuminosityBlockPrincipal;
  class RunPrincipal;

  class TestRunLumiSource : public InputSource {
  public:
    explicit TestRunLumiSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~TestRunLumiSource();

  private:

    virtual ItemType getNextItemType();
    virtual EventPrincipal* readEvent_();
    boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();

    // This vector holds 3 values representing (run, lumi, event)
    // repeated over and over again, in one vector.
    // Each set of 3 values is placed in the the auxiliary
    // object of the principal returned by a call
    // to readEvent_, readLuminosityBlock_, or readRun_.
    // Each set of 3 values is used in the order it appears in the vector.
    // (0, 0, 0) is a special value indicating the read
    // function should return a NULL value indicating last event,
    // last lumi, or last run.
    std::vector<int> runLumiEvent_;
    std::vector<int>::size_type currentIndex_;
    bool firstTime_;
  };
}
#endif
