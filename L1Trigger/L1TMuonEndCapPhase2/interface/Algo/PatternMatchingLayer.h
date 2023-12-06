#ifndef L1Trigger_L1TMuonEndCapPhase2_PatternMatchingLayer_h
#define L1Trigger_L1TMuonEndCapPhase2_PatternMatchingLayer_h

#include <array>
#include <vector>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2::algo {

    class PatternMatchingLayer {

        public:
            PatternMatchingLayer(const EMTFContext&);

            ~PatternMatchingLayer();

            void apply(
                    const std::vector<hitmap_t>&,
                    const bool&,
                    std::vector<road_collection_t>&
            ) const;

        private:
            const EMTFContext& context_;
    };

}

#endif  // L1Trigger_L1TMuonEndCapPhase2_PatternMatchingLayer_h not defined
