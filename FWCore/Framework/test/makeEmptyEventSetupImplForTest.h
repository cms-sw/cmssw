#ifndef FWCore_Framework_makeEmptyEventSetupImplForTest_h
#define FWCore_Framework_makeEmptyEventSetupImplForTest_h
#include "FWCore/Framework/interface/EventSetupImpl.h"

namespace edm {
  inline edm::EventSetupImpl makeEmptyEventSetupImplForTest() { return edm::EventSetupImpl(); }
}  // namespace edm
#endif  // FWCore_Framework_makeEmptyEventSetupImplForTest_h
