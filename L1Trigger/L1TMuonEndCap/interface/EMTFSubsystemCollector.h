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
  ) const;

  // RPC functions
  void cluster_rpc(const TriggerPrimitiveCollection& muon_primitives, TriggerPrimitiveCollection& clus_muon_primitives) const;

  // GEM functions
  void make_copad_gem(const TriggerPrimitiveCollection& muon_primitives, TriggerPrimitiveCollection& copad_muon_primitives) const;

  void cluster_gem(const TriggerPrimitiveCollection& muon_primitives, TriggerPrimitiveCollection& clus_muon_primitives) const;
};

#endif
