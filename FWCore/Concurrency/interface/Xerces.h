#ifndef FWCore_Concurrency_h
#define FWCore_Concurrency_h

namespace cms {
  namespace concurrency {
    // Use these in place of XMLPlatformUtils::Initialize and
    // XMLPlatformUtils::Terminate. They are guaranteed to work correctly with
    // a multithreaded environment.
    void xercesInitialize();
    void xercesTerminate();
  }
}

#endif // FWCore_Concurrency_h
