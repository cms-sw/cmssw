#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/util/PlatformUtils.hpp>
#include <mutex>
#include <thread>
XERCES_CPP_NAMESPACE_USE

namespace cms {
  namespace concurrency {
    namespace {
      std::mutex g_xerces_mutex;
    }
    // We need to make sure these these are serialized are only called by one
    // thread at the time.  Until the first Init succeeds the other threads
    // should not be allowed to proceed.  We also do not want two different
    // threads calling Init and Finalize at the same time. We therefore simply
    // use a global mutex to serialize everything.
    void xercesInitialize() {
      std::unique_lock<std::mutex> l(g_xerces_mutex);
      XMLPlatformUtils::Initialize();
    }

    void xercesTerminate() {
      std::unique_lock<std::mutex> l(g_xerces_mutex);
      XMLPlatformUtils::Terminate();
    }
  }
}

