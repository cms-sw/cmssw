#ifndef L1Trigger_L1TMuonEndCapPhase2_HitmapLayer_h
#define L1Trigger_L1TMuonEndCapPhase2_HitmapLayer_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"

namespace emtf::phase2::algo {

  class HitmapLayer {
  public:
    HitmapLayer(const EMTFContext&);

    ~HitmapLayer() = default;

    void apply(const segment_collection_t&, std::vector<hitmap_t>&) const;

  private:
    const EMTFContext& context_;
  };

}  // namespace emtf::phase2::algo

#endif  // L1Trigger_L1TMuonEndCapPhase2_HitmapLayer_h not defined
