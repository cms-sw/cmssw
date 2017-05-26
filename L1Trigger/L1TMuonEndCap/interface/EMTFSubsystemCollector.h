#ifndef L1TMuonEndCap_EMTFSubsystemCollector_h
#define L1TMuonEndCap_EMTFSubsystemCollector_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


// Forward declarations
namespace edm {
  class Event;
  class EDGetToken;
}


// Class declaration
class EMTFSubsystemCollector {
public:
  template<typename T>
  void extractPrimitives(
    T tag,
    const edm::Event& iEvent,
    const edm::EDGetToken& token,
    TriggerPrimitiveCollection& out
  );

};

#endif
