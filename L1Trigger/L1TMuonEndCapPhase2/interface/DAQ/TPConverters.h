#ifndef L1Trigger_L1TMuonEndCapPhase2_TPConverters_h
#define L1Trigger_L1TMuonEndCapPhase2_TPConverters_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"

namespace emtf::phase2 {

  class TPConverter {
  public:
    TPConverter() = default;

    virtual ~TPConverter() = default;

    virtual void convert(const TriggerPrimitive&, const TPInfo&, EMTFHit&) const = 0;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_TPConverters_h
