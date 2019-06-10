#ifndef L1TMuonEndCap_EMTFSubsystemCollector_h_experimental
#define L1TMuonEndCap_EMTFSubsystemCollector_h_experimental

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


// Forward declarations
namespace edm {
  class Event;
  class EDGetToken;
}


// The 'experimental' namespace is used to contain classes that have conflicts
// with the existing classes. The experimental classes should eventually
// replace the existing classes.

namespace experimental {

// Class declaration
class EMTFSubsystemCollector {
public:
  // For 1 input collection
  template<typename T>
  void extractPrimitives(
    T tag,
    const GeometryTranslator* tp_geom,
    const edm::Event& iEvent,
    const edm::EDGetToken& token,
    TriggerPrimitiveCollection& out
  ) const;

  // For 2 input collections
  template<typename T>
  void extractPrimitives(
    T tag,
    const GeometryTranslator* tp_geom,
    const edm::Event& iEvent,
    const edm::EDGetToken& token1,
    const edm::EDGetToken& token2,
    TriggerPrimitiveCollection& out
  ) const;
};

}  // namespace experimental

#endif
