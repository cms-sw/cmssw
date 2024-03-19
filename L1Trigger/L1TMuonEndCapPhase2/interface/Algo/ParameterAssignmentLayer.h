#ifndef L1Trigger_L1TMuonEndCapPhase2_ParameterAssignment_h
#define L1Trigger_L1TMuonEndCapPhase2_ParameterAssignment_h

#include <array>
#include <vector>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2::algo {

  class ParameterAssignmentLayer {
  public:
    ParameterAssignmentLayer(const EMTFContext&);

    ~ParameterAssignmentLayer() = default;

    void apply(const bool&, std::vector<track_t>&) const;

  private:
    const EMTFContext& context_;
  };

}  // namespace emtf::phase2::algo

#endif  // L1Trigger_L1TMuonEndCapPhase2_ParameterAssignment_h not defined
