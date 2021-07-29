#ifndef FWCore_PluginManager_ProblemTracker_h
#define FWCore_PluginManager_ProblemTracker_h

/**

A singleton for managing the plugin feedback function and
assert hook function. It is unclear how this will react in
a multithreaded application. 

It is possible that in the future this object will allow arbitrary
state object to be pushed in and pulled out so that when asserts
occur, good state information can be printed.

**/

namespace edm {
  class ProblemTracker {
  public:
    static ProblemTracker const* instance();
    ProblemTracker(const ProblemTracker&) = delete;

  private:
    ProblemTracker();
    ~ProblemTracker();
  };

  class AssertHandler {
  public:
    AssertHandler();
    AssertHandler(const AssertHandler&) = delete;
    ~AssertHandler();

  private:
    ProblemTracker const* pt_;
  };

}  // namespace edm

#endif
