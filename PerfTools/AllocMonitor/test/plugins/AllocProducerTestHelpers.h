#ifndef PerfTools_AllocMonitor_test_plugins_AllocProducerTestHelpers_h
#define PerfTools_AllocMonitor_test_plugins_AllocProducerTestHelpers_h
// Small helper shared by the PerfTools/AllocMonitor test-only EDProducers
// that exercise ModuleAllocMonitor with ExternalWork/Transformer modules

#include "DataFormats/TestObjects/interface/ThingCollection.h"

namespace allocMonTest {
  inline edmtest::ThingCollection makeThings(int nThings, int offset) {
    edmtest::ThingCollection things;
    for (int i = 0; i < nThings; ++i) {
      things.emplace_back(i + offset);
    }
    return things;
  }
}  // namespace allocMonTest

#endif
